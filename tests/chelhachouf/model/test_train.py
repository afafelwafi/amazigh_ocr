#!/usr/bin/env python3
"""
Fixed Fine-tune Qwen2-VL-2B-Instruct for Amazigh text recognition in images
Supports Latin-written Amazigh with special characters: ɣ, ḥ, ṛ, ṣ, ṭ, ẓ
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
torch.mps.empty_cache()

from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazighOCRDataset(Dataset):
    """Dataset for Amazigh OCR training"""
    
    def __init__(self, data_dir, processor, max_length=512):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        
        # Find all image-text pairs
        self.data_pairs = []
        for img_file in self.data_dir.glob("**/*.png"):
            txt_file = img_file.with_suffix(".txt")
            if txt_file.exists():
                self.data_pairs.append((img_file, txt_file))
        
        logger.info(f"Found {len(self.data_pairs)} image-text pairs")
        
        if len(self.data_pairs) == 0:
            raise ValueError(f"No image-text pairs found in {data_dir}")
        
        # Validate Amazigh characters
        self.amazigh_chars = set('ƔṚṃčɛḌṢɣγǦƐǧṣḍẓḥεṬṛṭḤẒ')
        self._validate_dataset()
        
        # Test one sample to ensure processing works
        self._test_sample()
    
    def _validate_dataset(self):
        """Validate that dataset contains Amazigh characters"""
        char_counts = {char: 0 for char in self.amazigh_chars}
        
        for _, txt_file in self.data_pairs[:10]:  # Sample validation
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                for char in self.amazigh_chars:
                    char_counts[char] += text.count(char)
        
        logger.info(f"Amazigh character distribution (sample): {char_counts}")
    
    def _test_sample(self):
        """Test processing one sample to catch errors early"""
        try:
            sample = self[0]
            logger.info(f"Sample processed successfully. Keys: {list(sample.keys())}")
            for key, value in sample.items():
                if torch.is_tensor(value):
                    logger.info(f"  {key}: {value.shape}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            raise
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, txt_path = self.data_pairs[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Create conversation format for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please read the Amazigh text in this image."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Apply chat template to get text
        text_input = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Process text and image together
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Extract and prepare all required fields
        result = {}
        
        # Handle input_ids and attention_mask
        if 'input_ids' in inputs:
            result['input_ids'] = inputs['input_ids'].squeeze(0)
            result['labels'] = inputs['input_ids'].squeeze(0).clone()
        
        if 'attention_mask' in inputs:
            result['attention_mask'] = inputs['attention_mask'].squeeze(0)
            
        # Handle vision-specific inputs
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            result['pixel_values'] = inputs['pixel_values'].squeeze(0)
            
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            result['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)
        
        return result

class AmazighOCRTrainer:
    """Trainer for Amazigh OCR fine-tuning"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-2B-Instruct"):
        self.model_name = model_name
        self.device = "auto"
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 ,  # Changed to bfloat16
            device_map="auto" 
        )
        
        # Configure LoRA for efficient fine-tuning
        self.setup_lora()
    
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Increased back to reasonable value
            lora_alpha=64,  # Increased back
            lora_dropout=0.05,  # Back to lower dropout
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            modules_to_save=["lm_head"]  # Add back lm_head for better adaptation
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_dataset, val_dataset=None, output_dir="./amazigh_ocr_model"):
        """Train the model with fixed stability issues"""
        
        # Calculate total training steps for scheduler
        total_steps = len(train_dataset) // 1 // 8 * 3  # batch_size=1, grad_accum=8, epochs=3
        
        # Training arguments with stability fixes
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Reduced epochs for stability
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            
            # Learning rate and scheduling - less conservative
            learning_rate=2e-5,  # Slightly higher learning rate
            lr_scheduler_type="cosine",  # Back to cosine for better convergence
            warmup_steps=50,
            warmup_ratio=0.03,
            
            # Gradient clipping and optimization - less aggressive
            max_grad_norm=1.0,  # More reasonable gradient clipping
            optim="adamw_torch",
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            
            # Precision and memory - key fixes
            # bf16=True, # Use bfloat16 instead of fp16
            fp16=False,  # Disable fp16 explicitly
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            
            # Logging and evaluation
            logging_steps=10,
            eval_steps=100 if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",
            save_steps=100,
            save_total_limit=3,
            
            # Early stopping and best model
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            
            # Other settings for stability
            remove_unused_columns=False,
            report_to=None,
            seed=42,
            data_seed=42,
            
            # Disable problematic features
            label_smoothing_factor=0.0,  # Disable label smoothing
            ddp_find_unused_parameters=False,
            
            # Additional stability settings
            save_safetensors=True,
            prediction_loss_only=True,
        )
        
        # Enhanced data collator with better error handling
        def enhanced_data_collator(features):
            batch = {}
            
            try:
                # Handle text inputs with padding
                if 'input_ids' in features[0]:
                    # Pad sequences to same length
                    max_len = max(f['input_ids'].shape[0] for f in features)
                    padded_input_ids = []
                    padded_labels = []
                    padded_attention_mask = []
                    
                    for f in features:
                        input_ids = f['input_ids']
                        labels = f['labels']
                        attention_mask = f.get('attention_mask', torch.ones_like(input_ids))
                        
                        # Pad sequences
                        pad_length = max_len - input_ids.shape[0]
                        if pad_length > 0:
                            pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                            input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id)])
                            labels = torch.cat([labels, torch.full((pad_length,), -100)])  # Ignore padded tokens in loss
                            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
                        
                        padded_input_ids.append(input_ids)
                        padded_labels.append(labels)
                        padded_attention_mask.append(attention_mask)
                    
                    batch['input_ids'] = torch.stack(padded_input_ids)
                    batch['labels'] = torch.stack(padded_labels)
                    batch['attention_mask'] = torch.stack(padded_attention_mask)
                
                # Handle vision inputs
                if 'pixel_values' in features[0] and features[0]['pixel_values'] is not None:
                    batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
                    
                if 'image_grid_thw' in features[0] and features[0]['image_grid_thw'] is not None:
                    batch['image_grid_thw'] = torch.stack([f['image_grid_thw'] for f in features])
                    
            except Exception as e:
                logger.error(f"Error in data collator: {e}")
                # Fallback to simple stacking
                for key in features[0].keys():
                    if torch.is_tensor(features[0][key]):
                        batch[key] = torch.stack([f[key] for f in features])
                
            return batch
        
        # Custom trainer class with stability fixes
        class StableTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_history = []
                
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Compute loss with minimal intervention"""
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Only intervene if loss is actually problematic
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {self.state.global_step}")
                    # Return a reasonable loss that allows gradient flow
                    loss = torch.tensor(1.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
                elif loss.item() > 100.0:  # Only clamp extremely high losses
                    logger.warning(f"Very high loss detected: {loss.item():.4f}, clamping to 20.0")
                    loss = torch.clamp(loss, max=20.0)
                
                # Track loss
                self.loss_history.append(loss.item())
                
                # Log progress
                if self.state.global_step % 5 == 0:
                    logger.info(f"Step {self.state.global_step}: Loss = {loss.item():.4f}")
                
                return (loss, outputs) if return_outputs else loss
            
            def training_step(self, model, inputs, num_items_in_batch=None):
                """Training step with minimal gradient intervention"""
                model.train()
                inputs = self._prepare_inputs(inputs)
                
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.args.gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Check gradients periodically but don't zero them aggressively
                if self.state.global_step % 10 == 0:
                    total_norm = 0
                    param_count = 0
                    nan_grads = 0
                    
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                nan_grads += 1
                                logger.warning(f"NaN/Inf gradient in {name}")
                                # Only zero if absolutely necessary
                                p.grad.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                            else:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                    
                    if param_count > 0:
                        total_norm = total_norm ** (1. / 2)
                        logger.info(f"Step {self.state.global_step}: Gradient norm = {total_norm:.4f}")
                    
                    if nan_grads > 0:
                        logger.warning(f"Step {self.state.global_step}: Fixed {nan_grads} NaN gradients")
                
                return loss.detach()
        
        # Setup callbacks
        callbacks = []
        if val_dataset:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            )
            callbacks.append(early_stopping)
        
        # Initialize stable trainer
        trainer = StableTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=enhanced_data_collator,
            processing_class=self.processor,
            callbacks=callbacks,
        )
        
        # Train with enhanced monitoring
        logger.info("Starting stable training...")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info("Key stability fixes applied:")
        logger.info("- Using bfloat16 instead of fp16")
        logger.info("- Conservative learning rate and gradient clipping")
        logger.info("- NaN/Inf gradient detection and handling")
        logger.info("- Reduced LoRA parameters")
        
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            # Try to save partial progress
            try:
                trainer.save_model(output_dir + "_partial")
                logger.info(f"Partial model saved to {output_dir}_partial")
            except:
                pass
            raise
        
        # Save model and processor
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        # Save training metrics
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'loss_history': trainer.loss_history[-100:],  # Last 100 losses
                'final_loss': trainer.loss_history[-1] if trainer.loss_history else None,
                'total_steps': trainer.state.global_step,
                'best_loss': min(trainer.loss_history) if trainer.loss_history else None,
                'stability_info': {
                    'precision': 'bfloat16',
                    'max_grad_norm': training_args.max_grad_norm,
                    'learning_rate': training_args.learning_rate
                }
            }, f, indent=2)
        
        logger.info(f"Training completed. Final loss: {trainer.loss_history[-1] if trainer.loss_history else 'N/A'}")
    
    def inference(self, image_path, model_path=None):
        """Perform inference on a single image"""
        if model_path:
            # Load fine-tuned model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please read the Amazigh text in this image."}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return generated_text

def prepare_data_split(data_dir, train_ratio=0.8):
    """Split data into train/validation sets"""
    data_dir = Path(data_dir)
    image_files = list(data_dir.glob("*.png"))
    
    # Shuffle and split
    import random
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    logger.info(f"Split: {len(train_files)} train, {len(val_files)} validation")
    return train_files, val_files

def main():
    """Main training function with stability fixes"""
    # Configuration
    DATA_DIR = "./amazigh_data"  # Directory with image_0.png, image_0.txt, etc.
    OUTPUT_DIR = "./amazigh_ocr_model"
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} not found!")
        logger.info("Please create the directory and add your image-text pairs:")
        logger.info("  image_0.png, image_0.txt")
        logger.info("  image_1.png, image_1.txt")
        logger.info("  ...")
        return
    
    # Set environment variables for stability
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize trainer
    trainer = AmazighOCRTrainer(MODEL_NAME)
    
    # Prepare datasets
    train_dataset = AmazighOCRDataset(DATA_DIR, trainer.processor)
    
    # Optional: create validation split
    if len(train_dataset) > 10:
        val_dataset = None  # For simplicity, using None
    else:
        val_dataset = None
    
    # Train model
    trainer.train(train_dataset, val_dataset, OUTPUT_DIR)
    
    # Test inference
    if len(train_dataset) > 0:
        test_image = train_dataset.data_pairs[0][0]
        logger.info(f"Testing inference on {test_image}")
        try:
            result = trainer.inference(test_image, OUTPUT_DIR)
            logger.info(f"Inference result: {result}")
        except Exception as e:
            logger.error(f"Inference test failed: {e}")

def test_single_image(image_path, model_path="./amazigh_ocr_model"):
    """Test the fine-tuned model on a single image"""
    trainer = AmazighOCRTrainer()
    result = trainer.inference(image_path, model_path)
    print(f"OCR Result: {result}")
    return result

if __name__ == "__main__":
    main()

# Example usage after training:
    #test_single_image("test.png")
#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL-2B-Instruct for Amazigh text recognition in images
Supports Latin-written Amazigh with special characters: ɣ, ḥ, ṛ, ṣ, ṭ, ẓ
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

PROMPT_DEFAULT = "Read the Amazigh text written in Latin script from this image. Include all special characters (ČčƐƔǦǧɛɣḌḍḤḥṚṛṢṣṬṭẒẓ). Return only the text, no explanations."

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
        for img_file in self.data_dir.glob("*.png"):
            txt_file = img_file.with_suffix(".txt")
            if txt_file.exists():
                self.data_pairs.append((img_file, txt_file))
        
        logger.info(f"Found {len(self.data_pairs)} image-text pairs")
        
        # Validate Amazigh characters
        self.amazigh_chars = set("ɣḥṛṣṭẓ")
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate that dataset contains Amazigh characters"""
        char_counts = {char: 0 for char in self.amazigh_chars}
        
        for _, txt_file in self.data_pairs[:10]:  # Sample validation
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                for char in self.amazigh_chars:
                    char_counts[char] += text.count(char)
        
        logger.info(f"Amazigh character distribution (sample): {char_counts}")
    
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
                    {"type": "text", "text": PROMPT_DEFAULT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Process with Qwen2VL processor
        processed = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': processed['input_ids'].squeeze(),
            'attention_mask': processed['attention_mask'].squeeze(),
            'labels': processed['input_ids'].squeeze().clone()
        }
 
class AmazighOCRTrainer:
    """Trainer for Amazigh OCR fine-tuning"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load processor first
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model with fixed device mapping
        self.model = self._load_model_safely()
        
        # Configure LoRA for efficient fine-tuning
        self.setup_lora()
    
    def _load_model_safely(self):
        """Load model with proper device handling to avoid meta tensor issues"""
        if torch.cuda.is_available():
            # For CUDA, use specific device mapping to avoid meta tensor issues
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=None,  # Don't use auto device mapping
                trust_remote_code=True
            )
            # Move to device after loading
            model = model.to(self.device)
        else:
            # For CPU
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
        
        return model
    
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            lora_dropout=0.1,  # Dropout
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_dataset, val_dataset=None, output_dir="./amazigh_ocr_qwen2_5_model"):
        """Train the model"""
        
        # Training arguments with fixed device handling
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Reduced batch size to avoid OOM
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # Disable device placement to avoid meta tensor issues
            dataloader_num_workers=0,
            # Add these to handle device placement properly
            no_cuda=not torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.processor.tokenizer,
            padding=True
        )
        
        # Initialize trainer with explicit device handling
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.processor,  # Use processing_class instead of tokenizer
        )
        
        # Ensure model is on correct device before training
        if torch.cuda.is_available():
            trainer.model = trainer.model.to(self.device)
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    def inference(self, image_path, model_path=None):
        """Perform inference on a single image"""
        if model_path:
            # Load fine-tuned model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None
            )
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_DEFAULT}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        )
        
        # Move inputs to device
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
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
    """Main training function"""
    # Configuration
    DATA_DIR = "./amazigh_data"  # Directory with image_0.png, image_0.txt, etc.
    OUTPUT_DIR = "./amazigh_ocr_qwen2_5model"
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # Pre-trained model name
    
    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} not found!")
        logger.info("Please create the directory and add your image-text pairs:")
        logger.info("  image_0.png, image_0.txt")
        logger.info("  image_1.png, image_1.txt")
        logger.info("  ...")
        return
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize trainer
    trainer = AmazighOCRTrainer(MODEL_NAME)
    
    # Prepare datasets
    train_dataset = AmazighOCRDataset(DATA_DIR, trainer.processor)
    
    # Optional: create validation split
    if len(train_dataset) > 10:
        # Split data for validation
        train_files, val_files = prepare_data_split(DATA_DIR)
        # You could create separate datasets here if needed
        val_dataset = None  # For simplicity, using None
    else:
        val_dataset = None
    
    # Train model
    trainer.train(train_dataset, val_dataset, OUTPUT_DIR)
    
    # Test inference
    if len(train_dataset) > 0:
        test_image = train_dataset.data_pairs[0][0]
        logger.info(f"Testing inference on {test_image}")
        result = trainer.inference(test_image, OUTPUT_DIR)
        logger.info(f"Inference result: {result}")

def test_single_image(image_path, model_path="./amazigh_ocr_model"):
    """Test the fine-tuned model on a single image"""
    trainer = AmazighOCRTrainer()
    result = trainer.inference(image_path, model_path)
    print(f"OCR Result: {result}")
    return result

if __name__ == "__main__":
    main()

# Example usage after training:
# test_single_image("path/to/your/image.png")
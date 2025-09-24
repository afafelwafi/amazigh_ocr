import torch
from transformers import AutoModelForImageTextToText, TrainingArguments, Trainer, AutoProcessor,  BitsAndBytesConfig, Qwen2VLForConditionalGeneration,Qwen2VLProcessor
from datasets import Dataset, load_dataset
from PIL import Image
import json
import os
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any
import logging
from data import *



import torch
from PIL import Image

# Qwen2-VL specific fix
def fix_qwen2vl_generation(model, processor, messages, image):
    """
    Fixed generation for Qwen2-VL model to handle image token mismatch
    """
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    print(f"Generated text: {text}")
    
    # Qwen2-VL specific preprocessing - resize image to standard size
    if isinstance(image, Image.Image):
        # Resize to a standard size that Qwen2-VL handles well
        image = image.resize((448, 448))  # Common size for Qwen2-VL
    
    # Process inputs with specific parameters for Qwen2-VL
    inputs = processor(
        text=[text], 
        images=[image], 
        return_tensors="pt",
        padding=True  # Add padding
    )
    
    print(f"Input keys: {inputs.keys()}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        print(f"Image grid thw: {inputs['image_grid_thw']}")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Qwen2-VL specific generation parameters
    generation_config = {
        "max_new_tokens": 256,
        "do_sample": False,     # Greedy decoding for accuracy
        "repetition_penalty": 1.2,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "num_beams": 2,         # Light beam search
        "early_stopping": True,
    }
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_config)
        
        # Decode only the new tokens
        input_token_len = inputs['input_ids'].shape[1]
        new_tokens = generated_ids[:, input_token_len:]
        
        decoded_text = processor.batch_decode(
            new_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        return decoded_text
        
    except ValueError as e:
        if "Image features and image tokens do not match" in str(e):
            print("Trying alternative approach...")
            return fix_qwen2vl_alternative(model, processor, messages, image)
        else:
            raise e

def fix_qwen2vl_alternative(model, processor, messages, image):
    """
    Alternative approach for Qwen2-VL when standard method fails
    """
    # Try with different image preprocessing
    if isinstance(image, Image.Image):
        # Try different standard sizes
        for size in [(336, 336), (224, 224), (512, 512)]:
            try:
                print(f"Trying image size: {size}")
                resized_image = image.resize(size)
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = processor(
                    text=[text], 
                    images=[resized_image], 
                    return_tensors="pt"
                )
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                input_token_len = inputs['input_ids'].shape[1]
                new_tokens = generated_ids[:, input_token_len:]
                
                decoded_text = processor.batch_decode(
                    new_tokens, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
                print(f"Success with image size: {size}")
                return decoded_text
                
            except Exception as e:
                print(f"Failed with size {size}: {e}")
                continue
    
    raise ValueError("All alternative approaches failed")

# Usage


if __name__ == "__main__":
    model_path = "amazigh_ocr_model/checkpoint-100"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,  torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Changed to bfloat16
                device_map="auto" if torch.cuda.is_available() else None)
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    
    image_path = "test.png"
    image = Image.open(image_path).convert('RGB')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text":  "Please read the Amazigh text in this image. "}
            ]
        }
    ]
try:
    result = fix_qwen2vl_generation(model, processor, messages, image)
    print("Generated text:", result)
except Exception as e:
    print(f"Error: {e}")
    print("This might be a model-specific issue. Consider:")
    print("1. Using a different image resolution")
    print("2. Checking if the model was loaded correctly")
    print("3. Ensuring compatible transformers version")
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
from tqdm import tqdm
from collections import Counter
import random
import numpy as np
import glob
from config import DATASET, IMAGE_DIR, IMAGE_SIZE, REMOVE_CHARS, FONT_MAX_SIZE, FONT_PATH, PADDING, FONT_MIN_SIZE


# random different fonts, text length, text rotation, text size, background color, lighting, noise
class AmazighImageGenerator:
    def __init__(self, font_dir=None, enable_augmentations=True):
        self.dataset = load_dataset(DATASET)
        self.special_chars = self._extract_special_chars()
        self.enable_augmentations = enable_augmentations
        
        # Load multiple fonts if font directory is provided
        self.fonts = self._load_fonts(font_dir) if font_dir else [FONT_PATH]
        
        self._ensure_dirs()

    def _load_fonts(self, font_dir):
        """Load all font files from a directory"""
        if not os.path.exists(font_dir):
            print(f"Font directory {font_dir} not found. Using default font.")
            return [FONT_PATH]
        
        font_extensions = ['*.ttf', '*.otf', '*.TTF', '*.OTF']
        fonts = []
        for ext in font_extensions:
            fonts.extend(glob.glob(os.path.join(font_dir, ext)))
        
        if not fonts:
            print(f"No font files found in {font_dir}. Using default font.")
            return [FONT_PATH]
        
        print(f"Loaded {len(fonts)} fonts from {font_dir}")
        return fonts

    def _extract_special_chars(self):
        transcriptions = list(self.dataset["train"]['transcription']) + list(self.dataset["test"]['transcription'])
        chars = [c for c in "".join(transcriptions) if c.isalpha() and not c.isascii()]
        chars = [c for c in chars if c not in REMOVE_CHARS]
        chars = set(chars)
        return sorted({ch.lower() for ch in chars} | {ch.upper() for ch in chars})

    def _ensure_dirs(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(os.path.join(IMAGE_DIR, "chars"), exist_ok=True)
        os.makedirs(os.path.join(IMAGE_DIR, "lines"), exist_ok=True)

    def _get_random_colors(self):
        """Generate random background and foreground colors with good contrast"""
        bg = tuple(random.randint(0, 255) for _ in range(3))
        while True:
            fg = tuple(random.randint(0, 255) for _ in range(3))
            # Ensure good contrast
            contrast = sum(abs(b - f) for b, f in zip(bg, fg))
            if contrast > 300:
                return bg, fg

    def _get_random_font(self):
        """Select a random font from available fonts"""
        return random.choice(self.fonts)

    def _add_noise(self, img, noise_level=0.1):
        """Add random noise to the image"""
        if not self.enable_augmentations or random.random() > 0.3:
            return img
            
        img_array = np.array(img)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def _add_lighting_effects(self, img):
        """Add random lighting effects"""
        if not self.enable_augmentations or random.random() > 0.4:
            return img
            
        # Random brightness adjustment
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = random.uniform(0.7, 1.3)
            img = enhancer.enhance(brightness_factor)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            contrast_factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(contrast_factor)
            
        # Subtle blur effect occasionally
        if random.random() > 0.8:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
        return img

    def _get_random_rotation(self):
        """Get random rotation angle"""
        if not self.enable_augmentations or random.random() > 0.3:
            return 0
        # Small rotations to keep text readable
        return random.uniform(-15, 15)

    def _fit_text(self, text, font_path, image_size, max_font, min_font, padding):
        """Find the best font size that fits the text in the image"""
        width, height = image_size
        for font_size in range(max_font, min_font - 1, -2):
            try:
                font = ImageFont.truetype(font_path, font_size)
                bbox = font.getbbox(text)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Account for potential rotation by using diagonal
                diagonal = int((text_width**2 + text_height**2)**0.5)
                if diagonal + 2 * padding <= min(width, height):
                    return font, (text_width, text_height)
            except OSError:
                # Skip invalid fonts
                continue
                
        # If text doesn't fit or font is invalid, return min font and actual size
        try:
            font = ImageFont.truetype(font_path, min_font)
            bbox = font.getbbox(text)
            return font, (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except OSError:
            # Fallback to default font
            font = ImageFont.load_default()
            bbox = font.getbbox(text)
            return font, (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _draw_text_centered(self, img, text, font, fill, rotation=0):
        """Draw text centered on the image with optional rotation"""
        if rotation == 0:
            # Simple case - no rotation
            draw = ImageDraw.Draw(img)
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            img_width, img_height = img.size
            x = (img_width - text_width) // 2
            y = (img_height - text_height) // 2
            draw.text((x, y), text, fill=fill, font=font)
        else:
            # Create a temporary image for rotation
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Create temporary image with text
            temp_size = max(text_width, text_height) + 40  # Extra padding for rotation
            temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Draw text centered in temp image
            x = (temp_size - text_width) // 2
            y = (temp_size - text_height) // 2
            temp_draw.text((x, y), text, fill=fill, font=font)
            
            # Rotate the temp image
            rotated = temp_img.rotate(rotation, expand=False, fillcolor=(0, 0, 0, 0))
            
            # Paste onto main image
            img_width, img_height = img.size
            paste_x = (img_width - temp_size) // 2
            paste_y = (img_height - temp_size) // 2
            img.paste(rotated, (paste_x, paste_y), rotated)

    def _get_random_text_length(self, text, min_ratio=0.3, max_ratio=1.0):
        """Get a random substring of the text"""
        if not self.enable_augmentations or random.random() > 0.2:
            return text
            
        min_len = max(1, int(len(text) * min_ratio))
        max_len = len(text)
        
        if min_len >= max_len:
            return text
            
        target_len = random.randint(min_len, max_len)
        start_pos = random.randint(0, len(text) - target_len)
        return text[start_pos:start_pos + target_len]

    def generate_char_images(self):
        """Generate images for individual characters with augmentations"""
        for char in tqdm(self.special_chars, desc="Generating character images"):
            bg_color, fg_color = self._get_random_colors()
            font_path = self._get_random_font()
            rotation = self._get_random_rotation()
            
            # Create base image
            img = Image.new('RGB', IMAGE_SIZE, color=bg_color)
            
            # Fit text and draw
            font, _ = self._fit_text(char, font_path, IMAGE_SIZE, FONT_MAX_SIZE, FONT_MIN_SIZE, PADDING)
            self._draw_text_centered(img, char, font, fg_color, rotation)
            
            # Apply augmentations
            img = self._add_lighting_effects(img)
            img = self._add_noise(img)
            
            # Save image and label
            img_path = os.path.join(IMAGE_DIR, "chars", f"char_{ord(char)}.png")
            txt_path = os.path.join(IMAGE_DIR, "chars", f"char_{ord(char)}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(char + "\n")

    def generate_line_images(self):
        """Generate images for text lines with augmentations"""
        transcriptions = list(self.dataset["train"]['transcription']) + list(self.dataset["test"]['transcription'])
        filtered = [t for t in transcriptions if any(c in self.special_chars for c in t)]
        
        for i, original_text in tqdm(enumerate(filtered), desc="Generating line images", total=len(filtered)):
            # Randomly truncate text
            text = self._get_random_text_length(original_text)
            
            bg_color, fg_color = self._get_random_colors()
            font_path = self._get_random_font()
            rotation = self._get_random_rotation()
            
            # Create base image
            img = Image.new('RGB', IMAGE_SIZE, color=bg_color)
            
            # Fit text and draw
            font, (tw, th) = self._fit_text(text, font_path, IMAGE_SIZE, FONT_MAX_SIZE, FONT_MIN_SIZE, PADDING)
            self._draw_text_centered(img, text, font, fg_color, rotation)
            
            # Apply augmentations
            img = self._add_lighting_effects(img)
            # img = self._add_noise(img)
            
            # Save image and label
            img_path = os.path.join(IMAGE_DIR, "lines", f"line_{i}.png")
            txt_path = os.path.join(IMAGE_DIR, "lines", f"line_{i}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")

    def generate_augmented_dataset(self, num_variations=5):
        """Generate multiple variations of each text with different augmentations"""
        transcriptions = self.dataset["train"]['transcription'].to_list() + self.dataset["test"]['transcription'].to_list()
        filtered = [t for t in transcriptions if any(c in self.special_chars for c in t)]
        
        os.makedirs(os.path.join(IMAGE_DIR, "augmented"), exist_ok=True)
        
        total_images = len(filtered) * num_variations
        pbar = tqdm(total=total_images, desc="Generating augmented dataset")
        
        for i, original_text in enumerate(filtered):
            for var in range(num_variations):
                # Apply random text length variation
                text = self._get_random_text_length(original_text)
                
                bg_color, fg_color = self._get_random_colors()
                font_path = self._get_random_font()
                rotation = self._get_random_rotation()
                
                # Create base image
                img = Image.new('RGB', IMAGE_SIZE, color=bg_color)
                
                # Fit text and draw
                font, _ = self._fit_text(text, font_path, IMAGE_SIZE, FONT_MAX_SIZE, FONT_MIN_SIZE, PADDING)
                self._draw_text_centered(img, text, font, fg_color, rotation)
                
                # Apply augmentations
                img = self._add_lighting_effects(img)
                # img = self._add_noise(img)
                
                # Save with variation number
                img_path = os.path.join(IMAGE_DIR, "augmented", f"text_{i}_var_{var}.png")
                txt_path = os.path.join(IMAGE_DIR, "augmented", f"text_{i}_var_{var}.txt")
                img.save(img_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
                
                pbar.update(1)
        
        pbar.close()


if __name__ == "__main__":
    # Usage examples:
    
    # Basic usage with default font
    generator = AmazighImageGenerator()
    generator.generate_char_images()
    generator.generate_line_images()
    
    # Advanced usage with multiple fonts and augmentations
    # generator = AmazighImageGenerator(font_dir="/path/to/fonts", enable_augmentations=True)
    # generator.generate_augmented_dataset(num_variations=3)
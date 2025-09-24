from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
from collections import Counter
import random
from config import *

class AmazighImageGenerator:
    def __init__(self):
        self.dataset = load_dataset(DATASET)
        self.special_chars = self._extract_special_chars()
        self._ensure_dirs()

    def _extract_special_chars(self):
        transcriptions = self.dataset["train"]['transcription'] + self.dataset["test"]['transcription']
        chars = [c for c in "".join(transcriptions) if c.isalpha() and not c.isascii()]
        chars = [c for c in chars if c not in REMOVE_CHARS]
        chars = set(chars)
        return sorted({ch.lower() for ch in chars} | {ch.upper() for ch in chars})

    def _ensure_dirs(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(os.path.join(IMAGE_DIR, "chars"), exist_ok=True)
        os.makedirs(os.path.join(IMAGE_DIR, "lines"), exist_ok=True)

    def _get_random_colors(self):
        bg = tuple(random.randint(0, 255) for _ in range(3))
        while True:
            fg = tuple(random.randint(0, 255) for _ in range(3))
            if sum(abs(b - f) for b, f in zip(bg, fg)) > 300:
                return bg, fg

    def _fit_text(self, text, font_path, image_size, max_font, min_font, padding):
        width, height = image_size
        for font_size in range(max_font, min_font - 1, -2):
            font = ImageFont.truetype(font_path, font_size)
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if text_width + 2 * padding <= width and text_height + 2 * padding <= height:
                return font, (text_width, text_height)
        # If text doesn't fit, return min font and actual size
        font = ImageFont.truetype(font_path, min_font)
        bbox = font.getbbox(text)
        return font, (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _draw_text_centered(self, img, text, font, fill, padding):
        draw = ImageDraw.Draw(img)
        bbox = font.getbbox(text)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img_width, img_height = img.size
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
        draw.text((x, y), text, fill=fill, font=font)

    def generate_char_images(self):
        for char in tqdm(self.special_chars, desc="Generating character images"):
            bg_color, fg_color = self._get_random_colors()
            img = Image.new('RGB', IMAGE_SIZE, color=bg_color)
            font, _ = self._fit_text(char, FONT_PATH, IMAGE_SIZE, FONT_MAX_SIZE, FONT_MIN_SIZE, PADDING)
            self._draw_text_centered(img, char, font, fg_color, PADDING)
            img_path = os.path.join(IMAGE_DIR, "chars", f"char_{ord(char)}.png")
            txt_path = os.path.join(IMAGE_DIR, "chars", f"char_{ord(char)}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(char + "\n")

    def generate_line_images(self):
        transcriptions = self.dataset["train"]['transcription'] + self.dataset["test"]['transcription']
        filtered = [t for t in transcriptions if any(c in self.special_chars for c in t)]
        for i, text in tqdm(enumerate(filtered), desc="Generating line images", total=len(filtered)):
            bg_color, fg_color = self._get_random_colors()
            font, (tw, th) = self._fit_text(text, FONT_PATH, IMAGE_SIZE, FONT_MAX_SIZE, FONT_MIN_SIZE, PADDING)
            img = Image.new('RGB', IMAGE_SIZE, color=bg_color)
            self._draw_text_centered(img, text, font, fg_color, PADDING)
            img_path = os.path.join(IMAGE_DIR, "lines", f"line_{i}.png")
            txt_path = os.path.join(IMAGE_DIR, "lines", f"line_{i}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")

if __name__ == "__main__":
    generator = AmazighImageGenerator()
    generator.generate_char_images()
    generator.generate_line_images()

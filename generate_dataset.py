from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import random
from collections import Counter


DATASET = "TutlaytAI/moroccan_amazigh_asr"
data = load_dataset(DATASET)  
transcriptions_list = data["train"]['transcription']+data["test"]['transcription']
list_of_chars = list("".join(transcriptions_list))
list_of_chars.remove(' ')  # Remove spaces if they are not neede
# remove punctuation
list_of_chars = [c for c in list_of_chars if c.isalpha()]
# remove latin letters  
list_of_chars = [c for c in list_of_chars if not c.isascii()]
special_amazigh_chars = list(set(list_of_chars))
transcriptions_list_filtered = [text for text in transcriptions_list if any(c in special_amazigh_chars for c in text)]

def generate_amazigh_image_and_text(text, font_path, image_path, text_path):
    # Génère des couleurs aléatoires contrastées
    def get_random_colors():
        bg_color = tuple(random.randint(0, 255) for _ in range(3))
        while True:
            fill_color = tuple(random.randint(0, 255) for _ in range(3))
            # Contraste simple basé sur différence RGB
            if sum(abs(b - f) for b, f in zip(bg_color, fill_color)) > 300:
                break
        return bg_color, fill_color

    bg_color, fill_color = get_random_colors()

    # Crée une image avec couleur de fond aléatoire
    img = Image.new('RGB', (800, 200), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Charge la police
    try:
        font = ImageFont.truetype(font_path, 60)
    except Exception as e:
        raise ValueError(f"Could not load font: {e}")

    # Dessine le texte avec couleur de remplissage aléatoire
    draw.text((60, 60), text, fill=fill_color, font=font)

    # Crée les répertoires si besoin
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(text_path), exist_ok=True)

    # Sauvegarde image
    img.save(image_path)
    print(f"Saved image to {image_path}")

    # Sauvegarde texte
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved text to {text_path}")



if __name__ == "__main__":
    # Generate images and texts for each transcription
    print("generate images and texts")
    for i in tqdm(range(len(transcriptions_list_filtered))):
        generate_amazigh_image_and_text(
            transcriptions_list_filtered[i],
            font_path="NotoSans-Regular.ttf",  # or Amazigh Unicode.ttf
            image_path=f"amazigh_data/image_{i}.png",
            text_path=f"amazigh_data/image_{i}.txt",
        )
    





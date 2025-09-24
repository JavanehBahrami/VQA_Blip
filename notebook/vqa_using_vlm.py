"""VQA_using_VLM.ipynb
"""

from google.colab import files
uploaded = files.upload()

import os
os.listdir()

from huggingface_hub import login
login("Your_HuggingFace_Token")

!pip install transformers torch torchvision --quiet
!pip install pillow --quiet

# Import libraries
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# --- 1. Load processor and model ---
# This is a smaller public BLIP-2 model that fits free Colab GPU
model_name = "Salesforce/blip2-flan-t5-xl"

print("Loading processor and model. This may take 1-2 minutes...")
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto"   # Automatically uses GPU if available, else CPU
)
print("Model loaded ✅")

"""# Visual Question Answering"""

# Load image
image_path = "horses.jpg"
raw_image = Image.open(image_path).convert("RGB")

# Questions in English
questions_en = [
    "What do you see in the image?",
    "Look at the horses. What colors do you see?",
    "How many horses are in the image?",
    "Which horse is closest to the camera, and what color is it?",
    "What might the horses be doing after this moment or after running?",
]

for q_en in questions_en:
    # inference from model
    inputs = processor(raw_image, q_en, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        answer_en = processor.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n🟢 Question: {q_en}")
    print(f"🔵 Answer: {answer_en}")

import os
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from huggingface_hub import login

# -------------------------------
# 0. Hugging Face login (safe)
# -------------------------------
# Set your HF token as environment variable first:
# In terminal or Colab: export HF_TOKEN=your_token_here
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("⚠️ Warning: No HF_TOKEN found. Make sure you set it before running this script.")

# -------------------------------
# 1. Configuration
# -------------------------------
# Set paths relative to repo root
IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
IMAGE_NAME = "horses.jpg"
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)

MODEL_NAME = "Salesforce/blip2-flan-t5-xl"  # fits free Colab GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 2. Load processor and model
# -------------------------------
print("Loading processor and model...")
processor = Blip2Processor.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto" if DEVICE == "cuda" else None,
    use_auth_token=hf_token
)
print(f"Model loaded on {DEVICE} ✅")

# -------------------------------
# 3. Load image
# -------------------------------
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
raw_image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {IMAGE_PATH}, size: {raw_image.size}")

# -------------------------------
# 4. Define questions
# -------------------------------
questions = [
    "What do you see in the image?",
    "Look at the horses. What colors do you see?",
    "How many horses are in the image?",
    "Which horse is closest to the camera, and what color is it?",
    "What might the horses be doing after this moment?",
]

# -------------------------------
# 5. Run VQA
# -------------------------------
for question in questions:
    inputs = processor(raw_image, question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True)

    print("\n🟢 Question:", question)
    print("🔵 Answer:", answer)



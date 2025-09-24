# VQA_Blip
Visual Question Answering (VQA) using BLIP-2 for multimodal understanding of images. Supports generating detailed answers from images with questions in English (and optionally other languages).

## Project Goal

This project demonstrates the application of Visual Language Models (VLMs), specifically BLIP-2, for Visual Question Answering (VQA). The goal is to combine computer vision and language understanding to analyze images and answer natural language questions about them. Users can ask questions about an image, such as “What objects are in this scene?” and the model provides accurate answers.

This repository also serves as a practical example for integrating VLMs into projects, showcasing how visual and textual reasoning can be combined in a single model. It’s designed to work on GPU-enabled environments like Google Colab.

## Demo image

<p align="center">
  <img src="assets/Demo_VQA_VLM.jpg" alt="VQA VLM Demo" width="600"/>
</p>


## Root Structure
The directory structure of this repository is organized as follows:

``` bash
VQA_Blip/                <-- root folder
│
├── images/              <-- store example input images for testing
│   └── horses.jpg       <-- sample image
│
├── models/              <-- optional: pre-trained or custom model checkpoints
│   └── README.md        <-- instructions if you store models locally
│
├── notebooks/           <-- Colab or Jupyter notebooks
│   └── VQA_demo.ipynb   <-- step-by-step interactive demo
│
├── src/                 <-- source code for inference & utilities
│   ├── inference.py     <-- main script to run VQA on images
│   ├── utils.py         <-- optional helper functions (e.g., image preprocessing)
│   └── config.py        <-- optional: paths, model names, or parameters
│
├── requirements.txt     <-- all Python dependencies for easy installation
│
├── README.md            <-- repo description, installation instructions, usage, examples
│
└── .gitignore           <-- ignore files like __pycache__, large models, Colab checkpoints
```

## Installation

1. Before running the project, install the required Python packages:

```bash
pip install -r requirements.txt
```

requirements.txt should include at least:

```bash
torch
transformers
Pillow
accelerate
```

2. Generate a Hugging Face token:

Go to Hugging Face Settings → Access Tokens

Create a new token.

Set your token as an environment variable (Linux/Mac/Colab):

```bash
export HF_TOKEN="your_huggingface_token_here"
```

## Usage

Place your input image(s) inside the image_folder.

Open `inference.py` inside the `src` folder and edit/add questions you want to ask the model.

Run the script:

```bash
python inference.py
```

The script will:

1. Load the BLIP-2 model and processor.

2. ead the input image.

3. Preprocess the image and questions.

4. Generate answers from the model.

5. Display questions and answers in the console.

Example questions you can ask:

```python
questions_en = [
    "What do you see in the image?",
    "Look at the horses. What colors do you see?",
    "How many horses are in the image?",
    "Which horse is closest to the camera, and what color is it?",
    "What might the horses be doing after this moment or after running?",
]
```


> **Notes:**
> - For faster inference, use a GPU-enabled environment such as **Google Colab** (T4 or better).
> - You can find the Colab notebook in the `notebook/` folder (`VQA_using_VLM.ipynb`) to see the code and try it interactively.

You can extend the script to handle multiple images or more advanced queries.

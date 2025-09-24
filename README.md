# VQA_Blip
Visual Question Answering (VQA) using BLIP-2 for multimodal understanding of images. Supports generating detailed answers from images with questions in English (and optionally other languages).


# Root Structure
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


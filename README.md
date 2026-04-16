---
title: Scene Image Classifier
emoji: "🏔️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8080
---

# Intel Image Classification

Joel Sangura Nyongesa  
AIMS Senegal 2026

This project implements a complete image classification pipeline for the Intel
Image Classification dataset using two custom CNN models:

- a PyTorch model saved as `joel_model.pth`
- a TensorFlow model saved as `joel_model.keras`

The models are served with Flask through a web interface that lets the user:

- choose the framework from a combo box
- upload an image
- view the predicted scene class and confidence scores

## Deployment

Accepted deployment for the course submission:

- Hugging Face Spaces: [scene-image-classifier](https://huggingface.co/spaces/joe254h/scene-image-classifier)

## Dataset

Source:

- <https://www.kaggle.com/datasets/puneet6060/intel-image-classification>

Expected local layout:

```text
data/
  seg_train/
    seg_train/
      buildings/
      forest/
      glacier/
      mountain/
      sea/
      street/
  seg_test/
    seg_test/
      buildings/
      forest/
      glacier/
      mountain/
      sea/
      street/
```

## Project Files

```text
app.py
train.py
joel_model.pth
joel_model.keras
requirements.txt
Dockerfile
fly.toml
templates/index.html
static/
```

## Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Main runtime dependencies:

- Flask
- Gunicorn
- NumPy
- Pillow
- PyTorch
- torchvision
- TensorFlow

## Training

Training is selected with the `--model` command-line argument, as required by
the project brief.

Examples:

```bash
python train.py --model pytorch --data_dir data
python train.py --model tensorflow --data_dir data
```

Notebook-aligned runs:

```bash
python train.py --model pytorch --data_dir data --epochs 70 --batch_size 128
python train.py --model tensorflow --data_dir data --epochs 50 --batch_size 64
```

GPU behavior:

- PyTorch uses CUDA automatically when available.
- TensorFlow uses GPU automatically and enables mirrored execution when more
  than one GPU is detected.

Final training environment notes:

- The final PyTorch Kaggle notebook used a Tesla T4 GPU.
- The final TensorFlow Kaggle notebook detected 2 Tesla T4 GPUs.

## Model Architectures

### PyTorch model

- 4 convolutional blocks: 64 -> 128 -> 256 -> 512 filters
- Batch normalization after each convolution pair
- Max pooling and dropout through the feature extractor
- Global average pooling before the classifier
- Dense(256) -> Dense(6)
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR

### TensorFlow model

- 3 convolutional blocks: 32 -> 64 -> 128 filters
- Batch normalization in every block
- Max pooling in the first two blocks
- Global average pooling before the classifier
- Dense(256) -> Dense(6)
- Optimizer: Adam
- Callbacks: ReduceLROnPlateau and EarlyStopping

## Final Results

| Model | Best validation accuracy | Test accuracy |
| --- | ---: | ---: |
| PyTorch | 89.50% | 89.33% |
| TensorFlow | 94.65% | 90.67% |

## Run the Web App Locally

```bash
python app.py
```

Then open:

- <http://127.0.0.1:5000>

## Notes

- The deployed app loads the saved PyTorch and TensorFlow models directly.
- The web interface is implemented with Flask, HTML, CSS, and JavaScript.
- Hugging Face Spaces was used as the final accepted hosting target for this
  submission.

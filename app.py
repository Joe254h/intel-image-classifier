import os
import io
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

CLASSES   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE  = 150
MAX_BYTES = 50 * 1024 * 1024  # 50 MB

_pytorch_model    = None
_tensorflow_model = None


def load_pytorch():
    global _pytorch_model
    if _pytorch_model is None:
        import torch
        import torch.nn as nn

        class JoelCNN_PyTorch(nn.Module):
            def __init__(self, num_classes=6):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2), nn.Dropout2d(0.15)
                )
                self.block3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2), nn.Dropout2d(0.2)
                )
                self.block4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                    nn.Dropout2d(0.2)
                )
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                x = self.gap(x)
                return self.classifier(x)

        device = torch.device("cpu")
        model  = JoelCNN_PyTorch(num_classes=6)
        model.load_state_dict(torch.load("joel_model.pth", map_location=device))
        model.eval()
        _pytorch_model = (model, device)
        print("PyTorch model loaded.")
    return _pytorch_model


def load_tensorflow():
    global _tensorflow_model
    if _tensorflow_model is None:
        import tensorflow as tf
        _tensorflow_model = tf.keras.models.load_model("joel_model.keras")
        print("TensorFlow model loaded.")
    return _tensorflow_model


def preprocess_for_pytorch(image_bytes):
    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


def preprocess_for_tensorflow(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file        = request.files["image"]
    model_name  = request.form.get("model", "pytorch")
    image_bytes = file.read()

    if len(image_bytes) > MAX_BYTES:
        return jsonify({"error": "Image too large (max 50 MB)"}), 400

    try:
        if model_name == "pytorch":
            import torch
            model, device = load_pytorch()
            tensor = preprocess_for_pytorch(image_bytes).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        elif model_name == "tensorflow":
            model = load_tensorflow()
            arr   = preprocess_for_tensorflow(image_bytes)
            probs = np.array(model.predict(arr, verbose=0)[0])

        else:
            return jsonify({"error": "Unknown model"}), 400

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        all_probs  = {cls: round(float(p) * 100, 2)
                      for cls, p in zip(CLASSES, probs)}

        return jsonify({
            "prediction": pred_class,
            "confidence": round(confidence, 2),
            "all_probs":  all_probs,
            "model_used": model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
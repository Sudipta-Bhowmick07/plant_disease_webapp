import os
import io
import json
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------ CONFIG ------------------

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "mobilenetv2_plant_disease.pth"
IMG_SIZE = 224

# ⚠️ IMPORTANT:
# Replace this list with the exact class_names you used in Colab (same order!)
class_names = [
    # EXAMPLE – REPLACE with your real list:
   'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    # ... paste all 38 names here in order ...
]

# ------------------ APP & MODEL SETUP ------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Albumentations transform (same style as training)
pred_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

def load_model():
    """Create MobileNetV2 architecture and load trained weights."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# ------------------ HELPER FUNCTIONS ------------------

def read_image(file_storage):
    """Read uploaded image file (Flask FileStorage) into RGB numpy array."""
    image_stream = file_storage.read()
    image = Image.open(io.BytesIO(image_stream)).convert("RGB")
    image = np.array(image)
    return image

def predict_image(image_np):
    """Run model prediction on a RGB numpy image."""
    # Albumentations expects dict
    aug = pred_tf(image=image_np)
    tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Sort top-3 predictions
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [
        {
            "class": class_names[int(i)],
            "confidence": float(probs[int(i)])
        }
        for i in top3_idx
    ]
    return pred_class, confidence, top3

# ------------------ ROUTES ------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"leaf_{timestamp}.png"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # We need file twice: once to read, once to save → reopen stream
            file.stream.seek(0)
            image_np = read_image(file)

            # Save a display copy (BGR→RGB handled by PIL already)
            Image.fromarray(image_np).save(save_path)

            # Predict
            pred_class, confidence, top3 = predict_image(image_np)

            return render_template(
                "index.html",
                uploaded_image=url_for("static", filename=f"uploads/{filename}"),
                prediction=pred_class,
                confidence=round(confidence, 3),
                top3=top3
            )

    # GET
    return render_template("index.html", uploaded_image=None, prediction=None, confidence=None, top3=None)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    app.run(debug=True)

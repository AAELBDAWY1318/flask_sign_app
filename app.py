from flask import Flask, request, jsonify
import torch
import os
import cv2
import numpy as np
from collections import Counter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load YOLOv5 model (from file)
try:
    model = torch.load('best.pt', map_location=torch.device('cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Image preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Failed to load image"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, None

def predict_from_image(image_path):
    img, error = preprocess_image(image_path)
    if error:
        return {"error": error}
    results = model(img)
    detections = results.pandas().xyxy[0]
    if detections.empty:
        return {"error": "No detections"}
    names = detections['name'].tolist()
    top_prediction = Counter(names).most_common(1)[0][0]
    return {"prediction": top_prediction, "detections": names}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)
    result = predict_from_image(image_path)
    return jsonify(result)

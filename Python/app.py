from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

print("Chargement du modèle YOLO...")
model = YOLO('yolov8n.pt')
print("Modèle chargé")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect():
    # لازم يكون فما 4 فراغات بالظبط في أول كل سطر داخل الـ function
    img_bytes = request.data
    
    if not img_bytes:
        return jsonify({'error': 'No image'}), 400

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    results = model(img, conf=0.3) 
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == 'person':
                person_detected = True
                break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, img)

    return jsonify({
        'person_detected': person_detected,
        'image': filename
    })

@app.route('/')
def home():
    return "Serveur YOLO opérationnel!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
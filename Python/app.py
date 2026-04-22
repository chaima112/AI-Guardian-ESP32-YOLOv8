from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO("best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_data = request.data
        if not image_data:
            return jsonify({"detected": False}), 400

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img, conf=0.25, verbose=False)

        is_person_found = False
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                is_person_found = True
                break

        print(f"Résultat de détection : {is_person_found}")
        return "true" if is_person_found else "false"

    except Exception as e:
        print(f"Erreur système : {e}")
        return "false", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

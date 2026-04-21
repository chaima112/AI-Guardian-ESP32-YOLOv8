from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# ============================================
# CHARGER LE MODÈLE best.pt (ton modèle entraîné)
# ============================================
model_path = "best.pt"

if not os.path.exists(model_path):
    print(f" Modèle {model_path} non trouvé!")
    print("   Assure-toi que best.pt est dans le même dossier")
    exit(1)

print(" Chargement du modèle YOLO...")
model = YOLO(model_path)
print(" Modèle chargé avec succès")

# ============================================
# ROUTE DE TEST
# ============================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "Serveur de détection d'intrusion",
        "model": "YOLOv8 (best.pt)"
    })

# ============================================
# ROUTE DE SANTÉ
# ============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

# ============================================
# ROUTE DE DÉTECTION
# ============================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Vérifier la requête
        if not request.json or 'image' not in request.json:
            return jsonify({"status": "error", "message": "Aucune image reçue"}), 400
        
        # Récupérer l'image en base64
        image_base64 = request.json['image']
        
        # Décoder l'image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"status": "error", "message": "Image invalide"}), 400
        
        # Inférence YOLO avec ton modèle best.pt
        results = model(img, conf=0.5)
        
        # Vérifier les détections
        if len(results[0].boxes) > 0:
            confidence = float(results[0].boxes[0].conf[0])
            return jsonify({
                "status": "alert",
                "confidence": confidence,
                "message": "Intrusion détectée"
            })
        
        return jsonify({
            "status": "normal",
            "message": "Aucune intrusion"
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================
# LANCEMENT DU SERVEUR
# ============================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print(" SERVEUR DE DÉTECTION D'INTRUSION")
    print("="*50)
    print(f" Modèle: {model_path}")
    print(f" http://0.0.0.0:5000")
    print(f"Routes disponibles:")
    print(f"   GET  /        - Vérifier le serveur")
    print(f"   GET  /health  - Health check")
    print(f"   POST /detect  - Détection d'intrusion")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

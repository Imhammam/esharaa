# app.py — Flask backend: يستقبل صورة من المتصفح ويرجع الحرف المتوقّع
from flask import Flask, request, jsonify, send_from_directory
import base64, json
from pathlib import Path
import numpy as np
import cv2, joblib, mediapipe as mp

# ==== تحميل المودِل والليبلات ====
MODEL_PATH = Path("model.pkl")
LE_PATH    = Path("label_encoder.pkl")
MAP_PATH   = Path("labels_map.json")

pipe = joblib.load(MODEL_PATH)
le   = joblib.load(LE_PATH)

LABELS_MAP = {}
if MAP_PATH.exists():
    try:
        LABELS_MAP = json.loads(MAP_PATH.read_text(encoding="utf-8"))["key_to_ar"]
    except Exception:
        LABELS_MAP = {}

def norm(name: str) -> str:
    s = (name or "").lower().strip()
    return {
        "shen":"sheen","wow":"waw","gaf":"qaf","haaa":"haa","alef":"alif",
        "taa":"ta","thaa":"tha","toot":"taa_marbouta","thaaa_heavy":"thaa_heavy",
    }.get(s, s)

def top1_prob(feat):
    proba = pipe.predict_proba([feat])[0]
    idx = int(np.argmax(proba))
    return le.inverse_transform([idx])[0], float(proba[idx])

# ==== mediapipe hands (جلسة واحدة لإعادة الاستخدام) ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,        # أسرع
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(img_bgr):
    """يرجع (features, hand_landmarks) أو (None, None)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None, None
    lm = res.multi_hand_landmarks[0].landmark
    pts = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)
    pts -= pts[0]  # مرجع عند المعصم
    scale = np.linalg.norm(pts[5,:2] - pts[17,:2]) + 1e-6
    pts /= scale
    return pts.flatten(), res.multi_hand_landmarks[0]

# ==== Flask app ====
app = Flask(__name__, static_url_path="", static_folder=".")

@app.route("/")
def index():
    # يقدّم index.html من نفس المجلد
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    data_url = data.get("image")
    if not data_url:
        return jsonify({"error":"no image"}), 400

    try:
        # data:image/jpeg;base64,xxxx
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error":"bad image"}), 400

        # (اختياري) تصغير لأداء أفضل
        if img.shape[1] > 640:
            scale = 640.0 / img.shape[1]
            img = cv2.resize(img, (640, int(img.shape[0]*scale)))

        feat, hand = extract_features(img)
        if feat is None:
            return jsonify({"nohand": True})

        label, prob = top1_prob(feat)
        key = norm(label)
        # الأحرف: ترجمة للعربي، أما الإشارات الوظيفية خليه بدون حرف عربي
        arabic = None
        if key not in ("space", "delete", "clear"):
            arabic = LABELS_MAP.get(key, LABELS_MAP.get(label, label))

        return jsonify({
            "label": key,         # ex: 'alif', 'space', 'delete', ...
            "prob":  float(prob),
            "arabic": arabic      # ex: 'ا' أو None لو وظيفة
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # شغّلي السيرفر
    app.run(host="127.0.0.1", port=5000, debug=True)

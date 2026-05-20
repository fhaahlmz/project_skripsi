# ============================================================
# SPK DETEKSI AWAL LESI KULIT
# Flask Deployment | CNN (MobileNetV2) + SVM
# ============================================================

from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import pickle
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from werkzeug.utils import secure_filename

# ============================================================
# KONFIGURASI
# ============================================================

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
IMG_SIZE = 224

# Pastikan folder upload ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("Memuat model...")

# Rename file model tanpa spasi agar aman di deployment
feature_extractor = load_model('CNN_feature_extractor.h5')

with open('svm_mobilenet_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

print("Model siap! ✓")

# ============================================================
# FUNGSI HELPER
# ============================================================

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features_cnn(img):
    """
    Ekstraksi fitur menggunakan MobileNetV2 fine-tuned
    """

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess_input(img.astype(np.float32))

    img = np.expand_dims(img, axis=0)

    features = feature_extractor.predict(img, verbose=0)

    return features.flatten()


def predict_image(img_path):
    """
    Prediksi lesi kulit dari gambar
    """

    img = cv2.imread(img_path)

    if img is None:
        return "Gambar tidak valid", None, None

    h, w = img.shape[:2]

    if h < 30 or w < 30:
        return "Gambar terlalu kecil", None, None

    # Ekstraksi fitur CNN
    features = extract_features_cnn(img).reshape(1, -1)

    # Prediksi SVM
    prob = svm_model.predict_proba(features)[0]
    pred = svm_model.predict(features)[0]

    confidence = np.max(prob) * 100

    # Level keyakinan
    if confidence < 55:
        return "Tidak dapat ditentukan", round(confidence, 2), "Sangat Tidak Yakin"

    elif confidence < 60:
        level = "Kurang Yakin"

    elif confidence < 75:
        level = "Cukup Yakin"

    elif confidence < 90:
        level = "Yakin"

    else:
        level = "Sangat Yakin"

    return pred, round(confidence, 2), level


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Cek file
    if 'file' not in request.files:
        return jsonify({
            'error': 'Tidak ada file yang diupload'
        }), 400

    file = request.files['file']

    # Cek nama file kosong
    if file.filename == '':
        return jsonify({
            'error': 'Tidak ada file yang dipilih'
        }), 400

    # Validasi format file
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Format tidak didukung. Gunakan JPG, PNG, atau WEBP'
        }), 400

    # Simpan file
    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    file.save(filepath)

    # Prediksi gambar
    result, confidence, level = predict_image(filepath)

    if confidence is None:
        return jsonify({
            'error': result
        }), 400

    return jsonify({
        'result': result,
        'confidence': confidence,
        'level': level,
        'image_url': f'/static/uploads/{filename}'
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )
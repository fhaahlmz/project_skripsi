# ============================================================
# SPK DETEKSI AWAL LESI KULIT
# Flask Deployment | CNN (MobileNetV2) + SVM
# + Active Learning: Feedback & Retraining
# ============================================================

from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import pickle
import os
import shutil
import threading
import json
import datetime

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from werkzeug.utils import secure_filename

# ============================================================
# KONFIGURASI
# ============================================================

app = Flask(__name__)

app.config['UPLOAD_FOLDER']    = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
IMG_SIZE           = 224

# Folder data
DATA_TRAIN_DIR   = 'data/train'          # data training asli (base)
DATA_PENDING_DIR = 'data/pending'        # foto baru dari user (belum ditraining)
DATA_TRAINED_DIR = 'data/trained'        # foto dari pending yg sudah dipakai retrain

MODEL_SVM_PATH   = 'svm_mobilenet_model.pkl'
MODEL_BACKUP_PATH = 'svm_mobilenet_model_backup.pkl'
STATS_PATH       = 'data/stats.json'

# Pastikan semua folder ada
for _dir in [
    app.config['UPLOAD_FOLDER'],
    f'{DATA_PENDING_DIR}/benign',
    f'{DATA_PENDING_DIR}/malignant',
    f'{DATA_PENDING_DIR}/unlabeled',   # foto yg user tidak tahu labelnya
    f'{DATA_TRAINED_DIR}/benign',
    f'{DATA_TRAINED_DIR}/malignant',
]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("Memuat model...")

feature_extractor = load_model('CNN_feature_extractor.h5')

with open(MODEL_SVM_PATH, 'rb') as f:
    svm_model = pickle.load(f)

# Lock untuk akses model secara thread-safe saat retraining
model_lock = threading.Lock()

# Status retraining
retrain_status = {
    'state':   'idle',   # idle | running | done | error
    'message': '',
    'progress': 0,
}

print("Model siap!")

# ============================================================
# HELPER: STATISTIK
# ============================================================

def _load_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as f:
            return json.load(f)
    return {
        'total_feedback': 0,
        'pending_benign':    0,
        'pending_malignant': 0,
        'total_retrain':     0,
        'last_retrain':      None,
    }


def _save_stats(stats):
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)


def _count_pending():
    def _c(folder):
        if not os.path.exists(folder):
            return 0
        return len([f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    b  = _c(f'{DATA_PENDING_DIR}/benign')
    m  = _c(f'{DATA_PENDING_DIR}/malignant')
    u  = _c(f'{DATA_PENDING_DIR}/unlabeled')
    return b, m, u


def _count_train():
    def _count(folder):
        if not os.path.exists(folder):
            return 0
        return len([f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    b = _count(f'{DATA_TRAIN_DIR}/benign')
    m = _count(f'{DATA_TRAIN_DIR}/malignant')
    return b, m

# ============================================================
# HELPER: FITUR & PREDIKSI
# ============================================================

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features_cnn(img):
    """Ekstraksi fitur menggunakan MobileNetV2 fine-tuned"""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()


def predict_image(img_path):
    """
    Prediksi lesi kulit dari gambar.
    Mengembalikan: (label, confidence, level, prob_benign, prob_malignant)
    """
    img = cv2.imread(img_path)

    if img is None:
        return "Gambar tidak valid", None, None, None, None

    h, w = img.shape[:2]
    if h < 30 or w < 30:
        return "Gambar terlalu kecil", None, None, None, None

    # Ekstraksi fitur CNN
    features = extract_features_cnn(img).reshape(1, -1)

    # Prediksi SVM (dengan lock agar aman saat model sedang di-swap)
    with model_lock:
        prob    = svm_model.predict_proba(features)[0]
        pred    = svm_model.predict(features)[0]
        classes = list(svm_model.classes_)

    idx_benign    = classes.index('benign')    if 'benign'    in classes else 0
    idx_malignant = classes.index('malignant') if 'malignant' in classes else 1

    prob_benign    = round(float(prob[idx_benign])    * 100, 2)
    prob_malignant = round(float(prob[idx_malignant]) * 100, 2)
    confidence     = round(np.max(prob) * 100, 2)

    # -------------------------------------------------------
    # THRESHOLD KEPUTUSAN: sesuai arahan dosen pembimbing
    # Confidence < 80% → sistem tidak dapat memutuskan kelas
    # Confidence >= 80% → keputusan benign / malignant valid
    # -------------------------------------------------------
    if confidence < 80:
        return "Tidak dapat ditentukan", confidence, _confidence_level(confidence), prob_benign, prob_malignant

    return pred, confidence, _confidence_level(confidence), prob_benign, prob_malignant


def _confidence_level(confidence):
    """Label tingkat keyakinan berdasarkan nilai confidence."""
    if confidence < 60:
        return "Sangat Rendah"
    elif confidence < 70:
        return "Rendah"
    elif confidence < 80:
        return "Cukup"
    elif confidence < 90:
        return "Tinggi"
    else:
        return "Sangat Tinggi"

# ============================================================
# RETRAINING (berjalan di background thread)
# ============================================================

def _retrain_background():
    """
    Proses retraining model SVM menggunakan:
    - data/train/ (data training asli)
    - data/pending/ (foto baru yang sudah dikonfirmasi labelnya)
    Model baru menggantikan model lama secara atomik.
    Server tetap melayani request selama proses ini berjalan.
    """
    global svm_model

    retrain_status['state']    = 'running'
    retrain_status['message']  = 'Mengumpulkan data...'
    retrain_status['progress'] = 5

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC

        X, y = [], []

        # Gabungkan sumber data: train/ + pending/
        sources = [DATA_TRAIN_DIR, DATA_PENDING_DIR]

        for source in sources:
            for label in ['benign', 'malignant']:
                folder = os.path.join(source, label)
                if not os.path.exists(folder):
                    continue
                files = [f for f in os.listdir(folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                for fname in files:
                    fpath = os.path.join(folder, fname)
                    img = cv2.imread(fpath)
                    if img is None:
                        continue
                    feat = extract_features_cnn(img)
                    X.append(feat)
                    y.append(label)

        if len(X) < 10:
            retrain_status['state']   = 'error'
            retrain_status['message'] = f'Data tidak cukup untuk retrain (hanya {len(X)} gambar). Minimal 10 gambar.'
            return

        retrain_status['message']  = f'Mengekstraksi fitur dari {len(X)} gambar...'
        retrain_status['progress'] = 40

        X = np.array(X)
        y = np.array(y)

        retrain_status['message']  = 'Melatih model SVM baru...'
        retrain_status['progress'] = 60

        # Bangun pipeline yang sama dengan model asli
        new_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=min(150, X.shape[0] - 1, X.shape[1]))),
            ('svm',    SVC(kernel='rbf', probability=True, C=10, gamma='scale')),
        ])
        new_pipeline.fit(X, y)

        retrain_status['message']  = 'Menyimpan model baru...'
        retrain_status['progress'] = 85

        # Simpan model baru ke file sementara
        new_model_path = MODEL_SVM_PATH + '.new'
        with open(new_model_path, 'wb') as f:
            pickle.dump(new_pipeline, f)

        # Backup model lama
        shutil.copy2(MODEL_SVM_PATH, MODEL_BACKUP_PATH)

        # Swap model di memory secara atomik
        with model_lock:
            svm_model = new_pipeline

        # Ganti file model utama
        os.replace(new_model_path, MODEL_SVM_PATH)

        # Pindahkan data pending ke trained (sudah dipakai)
        for label in ['benign', 'malignant']:
            src_folder = os.path.join(DATA_PENDING_DIR, label)
            dst_folder = os.path.join(DATA_TRAINED_DIR, label)
            os.makedirs(dst_folder, exist_ok=True)
            for fname in os.listdir(src_folder):
                shutil.move(
                    os.path.join(src_folder, fname),
                    os.path.join(dst_folder, fname)
                )

        # Update statistik
        stats = _load_stats()
        stats['total_retrain'] += 1
        stats['last_retrain']   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        stats['pending_benign']    = 0
        stats['pending_malignant'] = 0
        _save_stats(stats)

        retrain_status['state']    = 'done'
        retrain_status['message']  = f'Retraining selesai! Model dilatih dengan {len(X)} gambar.'
        retrain_status['progress'] = 100

    except Exception as e:
        retrain_status['state']   = 'error'
        retrain_status['message'] = f'Error saat retraining: {str(e)}'

# ============================================================
# ROUTES — Prediksi
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Format tidak didukung. Gunakan JPG, PNG, atau WEBP'}), 400

    # Buat nama file unik berbasis timestamp agar aman dari karakter khusus
    # (misal: nama file dari Google sering punya spasi / karakter aneh)
    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.webp'):
        ext = '.jpg'   # fallback aman
    unique_name = f"upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    result, confidence, level, prob_benign, prob_malignant = predict_image(filepath)

    if confidence is None:
        return jsonify({'error': result}), 400

    return jsonify({
        'result':         result,
        'confidence':     confidence,
        'level':          level,
        'prob_benign':    prob_benign,
        'prob_malignant': prob_malignant,
        'image_url':      f'/static/uploads/{unique_name}',
        'filename':       unique_name,   # nama konsisten untuk endpoint /feedback
    })

# ============================================================
# ROUTES — Active Learning: Feedback dari user
# ============================================================

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Menerima konfirmasi / koreksi / pelabelan dari pengguna.
    - benign / malignant : foto dengan label diketahui → data/pending/<label>/
    - unlabeled          : foto tanpa label (user tidak tahu) → data/pending/unlabeled/
    """
    data     = request.get_json()
    filename = data.get('filename', '')
    label    = data.get('label', '').lower()

    VALID_LABELS = ('benign', 'malignant', 'unlabeled')
    if label not in VALID_LABELS:
        return jsonify({'error': f'Label tidak valid. Gunakan: {", ".join(VALID_LABELS)}.'}), 400

    src = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(src):
        return jsonify({'error': 'File tidak ditemukan di server.'}), 404

    # Salin ke folder pending yang sesuai
    dst_dir  = os.path.join(DATA_PENDING_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)
    dst_name = secure_filename(filename)

    # Hindari duplikat nama file
    dst_path = os.path.join(dst_dir, dst_name)
    if os.path.exists(dst_path):
        base, ext = os.path.splitext(dst_name)
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        dst_path = os.path.join(dst_dir, f'{base}_{ts}{ext}')

    shutil.copy2(src, dst_path)

    # Pesan balasan sesuai label
    if label == 'unlabeled':
        msg = 'Terima kasih! Foto disimpan untuk direview oleh admin dan akan dilabeli sebelum digunakan sebagai data latih.'
    else:
        msg = f'Terima kasih! Foto disimpan sebagai data {label} untuk training berikutnya.'

    # Update statistik
    stats = _load_stats()
    stats['total_feedback'] += 1
    pb, pm, pu = _count_pending()
    stats['pending_benign']    = pb
    stats['pending_malignant'] = pm
    stats['pending_unlabeled'] = pu
    _save_stats(stats)

    return jsonify({
        'message':       msg,
        'label':         label,
        'pending_total': pb + pm + pu,
    })


# ============================================================
# ROUTES — Retraining (hanya bisa diakses dari localhost / admin)
# ============================================================

def _is_localhost():
    """Cek apakah request berasal dari localhost (akses admin)."""
    return request.remote_addr in ('127.0.0.1', '::1', '::ffff:127.0.0.1')

@app.route('/retrain', methods=['POST'])
def retrain():
    """Memicu retraining model di background thread. Hanya bisa diakses dari localhost."""
    if not _is_localhost():
        return jsonify({
            'error': 'Akses ditolak. Fitur retraining hanya tersedia untuk administrator melalui localhost.'
        }), 403

    if retrain_status['state'] == 'running':
        return jsonify({'error': 'Retraining sedang berjalan. Tunggu sebentar.'}), 409

    pb, pm, pu = _count_pending()
    total_pending = pb + pm   # hanya yg berlabel yg bisa langsung dipakai retrain

    if total_pending == 0:
        return jsonify({'error': 'Belum ada data berlabel (benign/malignant). Upload dan konfirmasi foto terlebih dahulu.'}), 400

    # Reset status
    retrain_status['state']    = 'idle'
    retrain_status['message']  = 'Memulai proses retraining...'
    retrain_status['progress'] = 0

    thread = threading.Thread(target=_retrain_background, daemon=True)
    thread.start()

    return jsonify({
        'message': f'Retraining dimulai dengan {total_pending} data berlabel baru. Server tetap melayani request.',
        'pending_benign':    pb,
        'pending_malignant': pm,
        'pending_unlabeled': pu,
    })


@app.route('/retrain-status', methods=['GET'])
def get_retrain_status():
    """Cek status proses retraining."""
    pb, pm, pu = _count_pending()
    stats  = _load_stats()
    tb, tm = _count_train()

    return jsonify({
        'state':             retrain_status['state'],
        'message':           retrain_status['message'],
        'progress':          retrain_status['progress'],
        'pending_benign':    pb,
        'pending_malignant': pm,
        'pending_unlabeled': pu,
        'pending_total':     pb + pm + pu,
        'train_benign':      tb,
        'train_malignant':   tm,
        'total_feedback':    stats.get('total_feedback', 0),
        'total_retrain':     stats.get('total_retrain', 0),
        'last_retrain':      stats.get('last_retrain', 'Belum pernah'),
    })

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
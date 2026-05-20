import gradio as gr
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# LOAD MODEL
# ============================================================

IMG_SIZE = 224

feature_extractor = load_model("CNN_feature_extractor.h5")

with open("svm_mobilenet_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# ============================================================
# PREDIKSI
# ============================================================

def predict_skin_lesion(image):

    if image is None:
        return "Tidak ada gambar"

    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess_input(img.astype(np.float32))

    img = np.expand_dims(img, axis=0)

    features = feature_extractor.predict(img, verbose=0).flatten()

    features = features.reshape(1, -1)

    prob = svm_model.predict_proba(features)[0]

    pred = svm_model.predict(features)[0]

    confidence = np.max(prob) * 100

    if confidence < 55:
        level = "Sangat Tidak Yakin"

    elif confidence < 60:
        level = "Kurang Yakin"

    elif confidence < 75:
        level = "Cukup Yakin"

    elif confidence < 90:
        level = "Yakin"

    else:
        level = "Sangat Yakin"

    return f"""
    Hasil Prediksi : {pred}

    Confidence : {confidence:.2f}%

    Tingkat Keyakinan : {level}
    """

# ============================================================
# GRADIO UI
# ============================================================

interface = gr.Interface(
    fn=predict_skin_lesion,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="SPK Deteksi Awal Lesi Kulit",
    description="CNN MobileNetV2 + SVM"
)

interface.launch()
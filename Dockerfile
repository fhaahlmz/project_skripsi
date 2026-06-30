FROM python:3.11-slim

# Install dependencies sistem untuk OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file proyek
COPY . .

# Buat folder yang diperlukan
RUN mkdir -p static/uploads \
    data/pending/benign \
    data/pending/malignant \
    data/pending/unlabeled \
    data/trained/benign \
    data/trained/malignant

# HuggingFace Spaces menggunakan port 7860
EXPOSE 7860

# Jalankan dengan gunicorn
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "120", \
     "--workers", "1", \
     "--log-level", "info"]

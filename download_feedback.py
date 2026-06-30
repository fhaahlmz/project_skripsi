"""
Script Admin: Download data feedback dari HuggingFace Dataset ke localhost.
Jalankan di localhost sebelum melakukan retraining.

Cara pakai:
    python download_feedback.py

Hasil download disimpan ke: data/pending/
"""

import os
from huggingface_hub import HfApi, list_repo_files

# ============================================================
# KONFIGURASI — sesuaikan dengan milikmu
# ============================================================
HF_TOKEN      = os.environ.get('HF_TOKEN', '')   # atau isi langsung: 'hf_xxx...'
HF_DATASET_ID = 'fhalmz/dermascan-feedback'
DOWNLOAD_DIR  = 'data/pending'
# ============================================================

def download_feedback():
    if not HF_TOKEN:
        print('[ERROR] HF_TOKEN tidak ditemukan.')
        print('Set environment variable: $env:HF_TOKEN="hf_xxx..."')
        print('Atau edit script ini dan isi HF_TOKEN secara langsung.')
        return

    api = HfApi(token=HF_TOKEN)

    print(f'Mengambil daftar file dari dataset: {HF_DATASET_ID}')

    try:
        files = list(list_repo_files(HF_DATASET_ID, repo_type='dataset', token=HF_TOKEN))
    except Exception as e:
        print(f'[ERROR] Gagal akses dataset: {e}')
        return

    image_files = [f for f in files if f.startswith('images/') and
                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    if not image_files:
        print('Tidak ada foto baru di dataset.')
        return

    print(f'Ditemukan {len(image_files)} foto. Memulai download...\n')

    downloaded = 0
    skipped    = 0

    for file_path in image_files:
        # file_path contoh: images/benign/foto.jpg
        parts = file_path.split('/')
        if len(parts) < 3:
            continue

        label    = parts[1]   # benign / malignant / unlabeled
        filename = parts[2]

        local_dir  = os.path.join(DOWNLOAD_DIR, label)
        local_path = os.path.join(local_dir, filename)

        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(local_path):
            skipped += 1
            continue  # sudah ada, skip

        try:
            api.hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=file_path,
                repo_type='dataset',
                local_dir='.',
            )
            # Pindahkan ke folder yang tepat
            downloaded_path = file_path  # relative path sesuai repo
            if os.path.exists(downloaded_path):
                os.makedirs(local_dir, exist_ok=True)
                os.rename(downloaded_path, local_path)

            print(f'  ✅ [{label}] {filename}')
            downloaded += 1
        except Exception as e:
            print(f'  ❌ Gagal download {filename}: {e}')

    print(f'\nSelesai! {downloaded} foto baru, {skipped} dilewati (sudah ada).')
    print(f'Data tersimpan di: {DOWNLOAD_DIR}/')
    print('\nSelanjutnya: buka Panel Admin di localhost → klik Retrain')

if __name__ == '__main__':
    download_feedback()

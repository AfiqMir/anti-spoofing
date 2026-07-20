# Anti Spoofing Backend — Setup Lokal

Backend ini menjalankan inferensi model final dari notebook
`Master_Optuna_KFold_AntiSpoofing_v2_Documented savedrive freeze backbone.ipynb`.

Backend ini dipertahankan untuk pengembangan lokal dan sebagai loader checkpoint
bagi `export_onnx.py`. Website yang dideploy tidak memakai backend ini karena
inferensi final berjalan langsung di browser melalui ONNX Runtime Web.

## Ringkasan model final

- ConvNeXt-Tiny pretrained ImageNet-1k; backbone dibekukan saat training dan
  classifier-nya adalah `Linear(768 -> 6)`.
- Input: satu gambar RGB, resize 384×384, normalisasi mean/std ImageNet.
- Urutan kelas: `realperson`, `fake_unknown`, `fake_mask`, `fake_mannequin`,
  `fake_screen`, `fake_printed`.
- Dataset: 1.652 gambar sebelum deduplikasi; 1.474 gambar setelah 178 duplikat
  dihapus.
- Training: Stratified 5-fold, 15 epoch, batch size 32, weighted focal loss
  (`gamma=2.18`), AdamW (`lr=9.74e-4`, `weight_decay=1.97e-5`), dan
  CosineAnnealingLR.
- Macro F1 rata-rata 5-fold: `0.9034 ± 0.0149`.
- Checkpoint terbaik: `master_convnext_fold1.pth`, dengan Macro F1 `0.9295`.
  Evaluasi validation Fold 1 menghasilkan accuracy `0.9356` pada 295 sampel.

## 1. Siapkan file model

Kelima model hasil training sudah tersedia di folder `backend`, masing-masing
sebagai paket checkpoint PyTorch yang telah diekstrak:

```text
master_convnext_fold1.pth/
master_convnext_fold2.pth/
master_convnext_fold3.pth/
master_convnext_fold4.pth/
master_convnext_fold5.pth/
```

Setiap paket berisi `data.pkl`, metadata format, dan 182 tensor shard pada
subfolder `data`. Kelimanya adalah artefak hasil training yang valid; Fold 1
adalah model dengan skor validasi terbaik.

Perlu dibedakan antara **isi checkpoint yang diekstrak** dan **file arsip
checkpoint**. Loader pada `main.py` sudah mendukung keduanya. Secara default,
backend membaca folder `master_convnext_fold1.pth`, mengemas strukturnya ke
buffer ZIP64 sementara, lalu memuatnya dengan `torch.load`. Proses tersebut tidak
mengubah file sumber, bobot, atau model hasil training.

Jika file `best_convnext_model.pth` tersedia, backend akan memprioritaskan file
tersebut. Lokasi checkpoint juga bisa ditentukan melalui environment variable
`ANTI_SPOOF_MODEL_PATH`.

Pastikan `classes.json` tetap berada di folder yang sama. Struktur yang dibaca
backend:

```text
backend/
  main.py
  requirements.txt
  classes.json
  master_convnext_fold1.pth/
    master_convnext_fold1/
      data.pkl
      data/
      version
```

## 2. Install dependency

Disarankan memakai virtual environment Python 3.11 agar sesuai dengan versi
dependency yang dikunci pada `requirements.txt`:

```text
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Jalankan backend

```text
uvicorn main:app --reload --port 8000
```

Kalau berhasil, akan muncul log kira-kira:

```text
Urutan kelas dimuat: ['realperson', 'fake_unknown', 'fake_mask', 'fake_mannequin', 'fake_screen', 'fake_printed']
Model dimuat dari .../master_convnext_fold1.pth ke device cpu
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## 4. Tes cepat

Buka `http://localhost:8000/health` di browser — respons harus memuat status
`"ok"` beserta urutan kelas.

Atau tes dari terminal:

```text
curl -X POST http://localhost:8000/predict -F "file=@/path/ke/foto.jpg"
```

## Troubleshooting

- **Checkpoint tidak ditemukan** — pastikan folder `master_convnext_fold1.pth`
  tersedia di folder backend, atau arahkan `ANTI_SPOOF_MODEL_PATH` ke checkpoint
  lain.
- **Root arsip PyTorch tidak ditemukan** — folder checkpoint harus memiliki satu
  subfolder root yang berisi `data.pkl`, `data/`, dan metadata checkpoint.
- **Prediksi terlihat tidak masuk akal** — pastikan urutan `classes.json` persis
  seperti urutan kelas final di atas dan preprocessing tidak diubah.
- **Lambat di CPU** — ConvNeXt-Tiny di CPU untuk satu gambar dapat lebih lambat
  daripada di GPU; hal ini wajar untuk model sebesar ini.

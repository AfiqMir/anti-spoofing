---
title: Face Anti Spoofing
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
---

# Anti Spoofing Web

Website statis untuk demo anti-spoofing. Upload gambar, akses kamera, preprocessing,
inferensi ONNX, softmax, dan tampilan enam probabilitas seluruhnya berjalan di
browser pengguna. Dark mode digunakan secara default dan pilihan tema disimpan
secara lokal oleh browser.

## Deployment Static Space

Unggah **isi folder `web`** ke satu Hugging Face Space dengan SDK **Static**.
Struktur pada root Space harus tetap seperti berikut:

```text
README.md
index.html
anti_spoof_convnext_fold1.onnx
```

Tidak ada Space variable, secret, Docker, Gradio, atau backend yang perlu
dikonfigurasi. File ONNX berukuran sekitar 106 MB, sehingga pemuatan pertama
memerlukan waktu sesuai kecepatan internet pengguna. Setelah indikator berubah
menjadi `Model siap · browser`, upload dan kamera dapat digunakan.

## Model final

Informasi model mengacu pada notebook final
`Master_Optuna_KFold_AntiSpoofing_v2_Documented savedrive freeze backbone.ipynb`:

- Arsitektur: ConvNeXt-Tiny pretrained ImageNet-1k dengan backbone dibekukan dan
  classifier `Linear(768 -> 6)`.
- Checkpoint deployment: Fold 1.
- Input inferensi: satu gambar RGB yang di-resize ke 384×384 dan dinormalisasi
  menggunakan mean/std ImageNet.
- Urutan output: `realperson`, `fake_unknown`, `fake_mask`, `fake_mannequin`,
  `fake_screen`, `fake_printed`.
- Data: 1.652 gambar sebelum deduplikasi dan 1.474 gambar sesudah 178 duplikat
  dihapus.
- Training final: Stratified 5-fold, 15 epoch, batch size 32, weighted focal
  loss (`gamma=2.18`), AdamW (`lr=9.74e-4`, `weight_decay=1.97e-5`), dan
  CosineAnnealingLR.
- Hasil: Macro F1 rata-rata 5-fold `0.9034 ± 0.0149`. Fold terbaik adalah Fold 1
  dengan Macro F1 `0.9295`; evaluasi validation Fold 1 berisi 295 sampel dan
  menghasilkan accuracy `0.9356`.

Metrik eksperimen didokumentasikan di README ini sebagai referensi, tetapi tidak
ditampilkan pada halaman publik `index.html`.

## Menjalankan secara lokal

```powershell
cd web
python -m http.server 5500
```

Buka `http://localhost:5500`, lalu tunggu model selesai dimuat. Static
server diperlukan agar file model dan akses kamera bekerja konsisten; jangan
membuka `index.html` langsung melalui `file://`.

## Deployment Vercel

Folder ini juga dapat dideploy melalui Vercel. `.vercelignore` mencegah model
ONNX 106 MB ikut terunggah karena melampaui batas file paket Hobby. Saat halaman
dibuka dari domain selain localhost dan Static Space, model otomatis diambil
dari Static Space Hugging Face yang sudah aktif.

```powershell
cd web
npx vercel
npx vercel --prod
```

Tidak diperlukan framework, build command, environment variable, atau backend.

## Menghasilkan ulang ONNX

Dari root proyek, jalankan:

```powershell
backend\.venv\Scripts\python.exe backend\export_onnx.py
```

Script memuat checkpoint Fold 1, mengekspor model, memvalidasi struktur ONNX,
dan membandingkan output ONNX Runtime dengan PyTorch.

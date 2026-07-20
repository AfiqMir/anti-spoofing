# Face Anti-Spoofing

Versi deployment proyek ini berjalan sepenuhnya sebagai **Hugging Face Static
Space**. Model final Fold 1 telah dikonversi ke ONNX dan inferensi dijalankan
langsung di browser dengan ONNX Runtime Web. Website publik tidak memerlukan
server atau layanan komputasi terpisah.

- `web/` berisi halaman siap deploy dan model ONNX final.
- `backend/` tetap berisi FastAPI untuk pengembangan lokal, checkpoint hasil
  training, dan `export_onnx.py` untuk menghasilkan ulang model browser.
- Folder Fold 2–5 tetap merupakan artefak training final dan tidak dipakai untuk
  inferensi karena model deployment yang dipilih adalah Fold 1.

Lihat petunjuk deployment pada `web/README.md`.

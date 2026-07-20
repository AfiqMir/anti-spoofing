# Face Anti-Spoofing

Versi deployment proyek ini berjalan sepenuhnya sebagai **Hugging Face Static
Space**. Model final telah dikonversi ke ONNX dan inferensi dijalankan
langsung di browser dengan ONNX Runtime Web. Website publik tidak memerlukan
server atau layanan komputasi terpisah.

- `web/` berisi halaman siap deploy dan model ONNX final.
- `backend/` tetap berisi FastAPI untuk pengembangan lokal, checkpoint hasil
  training, dan `export_onnx.py` untuk menghasilkan ulang model browser.
- Folder Fold q–5 tetap merupakan artefak training final.

Lihat petunjuk deployment pada `web/README.md`.

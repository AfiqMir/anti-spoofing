"""
Anti Spoofing — Backend inference untuk model ConvNeXt anti-spoofing.

Arsitektur & preprocessing di file ini HARUS PERSIS SAMA dengan yang dipakai
di notebook training (Master_Optuna_KFold_AntiSpoofing.ipynb), karena beda
sedikit saja (ukuran gambar, normalisasi, urutan kelas) bisa membuat model
salah prediksi walau checkpoint-nya benar.

Cara pakai:
1. Pastikan folder checkpoint hasil training `master_convnext_fold1.pth`
   tersedia di folder backend. File arsip `.pth` biasa juga tetap didukung.
2. Taruh juga "classes.json" di folder yang sama
3. Install dependency: pip install -r requirements.txt
4. Jalankan: uvicorn main:app --reload --port 8000
5. Backend siap diakses di http://localhost:8000
"""

import json
import os
import tempfile
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Konfigurasi — HARUS sama persis dengan notebook training
# ---------------------------------------------------------------------------
IMG_SIZE = 384
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
BACKEND_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
ARCHIVE_MODEL_PATH = os.path.join(BACKEND_DIR, "best_convnext_model.pth")
EXTRACTED_MODEL_PATH = os.path.join(BACKEND_DIR, "master_convnext_fold1.pth")
WEB_PATH = os.environ.get("ANTI_SPOOF_WEB_PATH") or os.path.join(PROJECT_DIR, "web", "index.html")

# Bisa dioverride untuk deployment, misalnya:
# ANTI_SPOOF_MODEL_PATH=/path/ke/checkpoint.pth
MODEL_PATH = os.environ.get("ANTI_SPOOF_MODEL_PATH") or (
    ARCHIVE_MODEL_PATH if os.path.isfile(ARCHIVE_MODEL_PATH) else EXTRACTED_MODEL_PATH
)
CLASSES_PATH = os.path.join(BACKEND_DIR, "classes.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load urutan kelas — WAJIB dari file classes.json hasil export notebook,
# JANGAN ditulis ulang manual di sini supaya tidak ada risiko urutan salah.
# ---------------------------------------------------------------------------
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(
        f"'{CLASSES_PATH}' tidak ditemukan. Copy file classes.json hasil "
        f"export notebook (folder model_output) ke folder backend ini."
    )

with open(CLASSES_PATH) as f:
    CLASSES = json.load(f)

print(f"Urutan kelas dimuat: {CLASSES}")


# ---------------------------------------------------------------------------
# Arsitektur model — disalin persis dari notebook training
# ---------------------------------------------------------------------------
class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=6, pretrained=False):
        super(ConvNeXtModel, self).__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


# ---------------------------------------------------------------------------
# Preprocessing — disalin persis dari val_transform di notebook training
# (Resize -> Normalize ImageNet -> ToTensor, TANPA augmentasi)
# ---------------------------------------------------------------------------
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal decode gambar — pastikan file yang diupload adalah gambar valid")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = val_transform(image=img)["image"]
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Load model sekali saat startup
# ---------------------------------------------------------------------------
model = None


def _find_extracted_archive_root(checkpoint_dir: str) -> str:
    """Temukan root arsip PyTorch yang memiliki data.pkl di checkpoint hasil ekstraksi."""
    if os.path.isfile(os.path.join(checkpoint_dir, "data.pkl")):
        return checkpoint_dir

    candidates = [
        entry.path
        for entry in os.scandir(checkpoint_dir)
        if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "data.pkl"))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Folder checkpoint '{checkpoint_dir}' harus berisi tepat satu root "
            f"arsip PyTorch dengan file data.pkl; ditemukan {len(candidates)}."
        )
    return candidates[0]


def load_checkpoint_state_dict(checkpoint_path: str):
    """Muat state_dict dari file .pth atau folder hasil ekstraksi arsip torch.save."""
    if os.path.isfile(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device, weights_only=True)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' tidak ditemukan sebagai file atau folder."
        )

    archive_root = _find_extracted_archive_root(checkpoint_path)
    archive_base = os.path.dirname(archive_root)

    # torch.save memakai arsip ZIP64. Folder hasil ekstraksi dikemas sementara
    # tanpa kompresi, lalu langsung dibaca torch.load; bobot sumber tidak diubah.
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as buffer:
        with zipfile.ZipFile(
            buffer, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for current_dir, dirnames, filenames in os.walk(archive_root):
                dirnames.sort()
                filenames.sort()
                for filename in filenames:
                    source_path = os.path.join(current_dir, filename)
                    archive_name = os.path.relpath(source_path, archive_base).replace(os.sep, "/")
                    archive.write(source_path, archive_name)

        buffer.seek(0)
        return torch.load(buffer, map_location=device, weights_only=True)


def load_model():
    global model
    m = ConvNeXtModel(num_classes=len(CLASSES), pretrained=False).to(device)
    state_dict = load_checkpoint_state_dict(MODEL_PATH)
    m.load_state_dict(state_dict)
    m.eval()
    print(f"Model dimuat dari {MODEL_PATH} ke device {device}")
    return m


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Anti Spoofing API")

# Deployment memakai same-origin. Origin localhost tetap diizinkan untuk
# pengembangan ketika frontend dan backend dijalankan pada port berbeda.
cors_origins = os.environ.get(
    "ANTI_SPOOF_CORS_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/", include_in_schema=False)
def serve_frontend():
    if not os.path.isfile(WEB_PATH):
        raise HTTPException(status_code=404, detail="Frontend index.html tidak ditemukan")
    return FileResponse(WEB_PATH, media_type="text/html")


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "model belum dimuat",
        "device": str(device),
        "classes": CLASSES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diupload harus berupa gambar")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran gambar maksimal 10 MB")

    try:
        input_tensor = preprocess_image(image_bytes).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        scores = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        return {
            "prediction": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_classes": scores,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {e}")

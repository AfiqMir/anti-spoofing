"""Export model final Fold 1 ke ONNX untuk inferensi langsung di browser."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import main


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "web" / "anti_spoof_convnext_fold1.onnx"


def export_model() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = main.load_model().cpu().eval()
    torch.manual_seed(2026)
    sample = torch.rand(1, 3, main.IMG_SIZE, main.IMG_SIZE, dtype=torch.float32)

    with torch.inference_mode():
        expected = model(sample).numpy()

    torch.onnx.export(
        model,
        sample,
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    exported = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(exported)

    session = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"input": sample.numpy()})[0]
    max_difference = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Output ONNX berbeda dari PyTorch (maksimum selisih {max_difference:.8f})")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"ONNX valid: {OUTPUT_PATH}")
    print(f"Ukuran: {size_mb:.2f} MB")
    print(f"Maksimum selisih output PyTorch/ONNX: {max_difference:.8f}")


if __name__ == "__main__":
    export_model()

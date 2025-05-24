# create_onnx_model.py

"""
Script to download and convert the NVIDIA NeMo Hindi Conformer CTC ASR model
to ONNX format for optimized inference.
"""

import torch
from nemo.collections.asr.models import EncDecCTCModel
import os

# Optional: set environment variables for NeMo caching
os.environ["TORCH_HOME"] = "./.cache"

# Output path for ONNX model
ONNX_EXPORT_PATH = "models/asr_conformer_hi.onnx"

def export_model_to_onnx():
    print("Loading pre-trained NeMo model...")
    model = EncDecCTCModel.from_pretrained(model_name="stt_hi_conformer_ctc_medium")

    print("Model loaded. Switching to evaluation mode...")
    model.eval()

    print("Converting to TorchScript...")
    # Example dummy input for TorchScript conversion
    dummy_input = torch.randn(1, 64, 160)  # (batch, features, time steps)
    traced = torch.jit.trace(model, dummy_input)

    print("Exporting to ONNX...")
    torch.onnx.export(
        traced,
        dummy_input,
        ONNX_EXPORT_PATH,
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"ONNX model saved at: {ONNX_EXPORT_PATH}")

if __name__ == "__main__":
    export_model_to_onnx()

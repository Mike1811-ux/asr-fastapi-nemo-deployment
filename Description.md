# 📝 Description.md — Hindi ASR API Assignment

## ✅ Features Implemented

- ✔️ Downloaded and used the official NVIDIA NeMo `stt_hi_conformer_ctc_medium` model for Hindi ASR.
- ✔️ Converted the model to ONNX format using TorchScript and `torch.onnx.export`.
- ✔️ Built a FastAPI server with a `/transcribe` endpoint.
- ✔️ Supported `.wav` audio files with strict validation:
  - Must be 16kHz sample rate.
  - Must be 5–10 seconds duration.
  - Only accepts `.wav` format.
- ✔️ Implemented model inference pipeline using ONNX Runtime.
- ✔️ Integrated audio preprocessing (resampling, normalization, mono conversion).
- ✔️ Created a clean Dockerfile using `python:3.10-slim` with minimal image size.
- ✔️ Wrote modular, well-commented, and production-ready Python code.
- ✔️ Provided detailed `README.md` and this description to support review.

---

## 🚧 Challenges Faced

### 1. **Model Optimization Complexity**
   - NeMo models are not natively ONNX-exportable due to dynamic graph structure.
   - Solution: Used TorchScript tracing as an intermediate step before ONNX export.

### 2. **Preprocessing Pipeline Matching**
   - The ONNX model expects log mel filterbank features with specific normalization.
   - Solution: Replicated the featurizer behavior using NeMo’s `FilterbankFeatures` module.

### 3. **Audio Duration/Rate Handling**
   - Ensured rejection of invalid inputs at upload time using the `wave` module.
   - Auto-resampled any misaligned inputs to 16kHz.

---

## ❌ Limitations

- 🚫 **True async inference** not implemented due to ONNXRuntime’s synchronous API. FastAPI supports async endpoints, but the inference call is blocking.
- 🚫 **No automatic CI/CD or testing framework** included due to time constraints.

---

## 🛠️ How I’d Improve Further

- ✅ Use a queue-based async inference architecture with Celery + Redis to support non-blocking ASR.
- ✅ Add pytest-based tests and GitHub Actions for CI.
- ✅ Build a frontend to upload audio and see real-time results.
- ✅ Use quantization-aware training or ONNX quantization tools to shrink the model size.

---

## ℹ️ Assumptions

- The deployed model is inference-only and won’t be retrained.
- Input is Hindi speech only, and may fail for other languages.
- Model must be used in an environment with access to NVIDIA’s pretrained weights (i.e., internet access for initial download).

---

## ✅ Summary

The project implements a fully functional ASR API with containerized deployment, clean modular code, strict validation, and documentation. All core objectives were met, and the system is designed for scalability and further enhancement.

> Thank you for reviewing this submission!

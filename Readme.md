asr_fastapi_app/
├── app/
│   ├── main.py           # FastAPI server
│   ├── model.py          # ONNX model loader & inference
│   ├── utils.py          # File validation & audio preprocessing
│   └── config.py         # (Optional) configuration constants
├── create_onnx_model.py  # Script to export NeMo model to ONNX
├── Dockerfile            # Container spec
├── requirements.txt      # Dependency list
├── README.md             # This file
├── Description.md        # Submission write-up
└── test_audio.wav        # Sample audio file (optional)


# 🚀 Hindi ASR API — FastAPI + NVIDIA NeMo + ONNX

This project implements a production-ready Automatic Speech Recognition (ASR) API using [NVIDIA NeMo](https://developer.nvidia.com/nvidia-nemo) and FastAPI. The API supports Hindi audio transcription for `.wav` files of 5–10 seconds duration and 16kHz sample rate.

---

## 📌 Features

- ✅ REST API with `/transcribe` endpoint
- ✅ Accepts `.wav` files (5–10s @ 16kHz)
- ✅ Transcribes audio using NVIDIA NeMo's Conformer CTC model
- ✅ Optimized using ONNX for efficient inference
- ✅ Dockerized and ready for deployment

---

## 🛠️ Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/)
- [NVIDIA NeMo Toolkit](https://developer.nvidia.com/nvidia-nemo)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Docker](https://www.docker.com/)
- [Torchaudio](https://pytorch.org/audio/)

---

## 🧪 Example Usage

### ▶️ Transcription via `curl`

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.wav"

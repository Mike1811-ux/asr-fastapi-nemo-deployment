## 📁 Project Structure
asr_fastapi_app/
├── app/ # Main FastAPI application package
│ ├── main.py # FastAPI server and /transcribe endpoint
│ ├── model.py # ONNX model loading and inference logic
│ └── utils.py # Audio validation and preprocessing
├── create_onnx_model.py # Script to convert NeMo model to ONNX
├── models/ # (Not tracked in Git) Folder for ONNX model
│ └── asr_conformer_hi.onnx # Generated ONNX model (must run script)
├── test_audio.wav # Sample test audio (5s, 16kHz sine wave)
├── Dockerfile # Docker container configuration
├── requirements.txt # Project dependencies
├── README.md # Project overview and setup instructions
└── Description.md # Documentation on features, issues, limitations


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

## 🧠 ONNX Model Export

The ONNX model file (`models/asr_conformer_hi.onnx`) is **not included** in this repository due to its size.

To generate the model locally, run the following script:

```bash
python create_onnx_model.py

## 🧪 Example Usage

### ▶️ Transcription via `curl`

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.wav"
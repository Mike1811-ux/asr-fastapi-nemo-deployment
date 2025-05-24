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
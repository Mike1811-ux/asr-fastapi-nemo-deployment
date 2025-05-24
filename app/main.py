# app/main.py

"""
FastAPI application to expose an endpoint for speech-to-text transcription using
an optimized ONNX model and NVIDIA NeMo-based ASR pipeline.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.utils import validate_audio, load_audio_tensor
from app.model import ASRModel

app = FastAPI(
    title="Hindi ASR API",
    description="Transcribes 16kHz Hindi audio (5–10s) using an optimized NeMo Conformer model",
    version="1.0.0",
)

# Optional: Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once globally
asr_model = ASRModel()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts a .wav file and returns the transcribed Hindi text.
    """
    try:
        # Validate and load audio
        audio_bytes = validate_audio(file)
        waveform_np, sr = load_audio_tensor(audio_bytes)

        # Inference
        transcript = asr_model.infer(waveform_np, sr)
        return JSONResponse(content={"transcript": transcript})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

# app/utils.py

"""
Utility functions for validating and preprocessing uploaded audio files.
Ensures .wav format
Verifies 16kHz sampling rate
Validates 5–10 sec duration
Converts audio to (1, time) NumPy array
"""

import io
import wave
from fastapi import UploadFile, HTTPException
import torchaudio
import torch
import numpy as np

MIN_DURATION = 5  # seconds
MAX_DURATION = 10  # seconds
EXPECTED_SAMPLE_RATE = 16000

def validate_audio(file: UploadFile) -> bytes:
    """
    Validates if the uploaded file is a valid .wav audio file.
    Args:
        file (UploadFile): Uploaded audio file.
    Returns:
        bytes: Raw audio file bytes if valid.
    Raises:
        HTTPException: If file is not valid.
    """
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    content = file.file.read()
    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            duration = wav_file.getnframes() / sample_rate

            if sample_rate != EXPECTED_SAMPLE_RATE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio sample rate must be 16kHz. Found: {sample_rate} Hz"
                )
            if not (MIN_DURATION <= duration <= MAX_DURATION):
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration must be between 5 and 10 seconds. Found: {duration:.2f}s"
                )
    except wave.Error:
        raise HTTPException(status_code=400, detail="Invalid WAV file format.")

    return content

from typing import Tuple

def load_audio_tensor(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Loads audio bytes into a normalized waveform tensor.
    Args:
        audio_bytes (bytes): Raw WAV file content.
    Returns:
        (np.ndarray, int): Tuple of waveform (1, time) as float32 numpy and sample rate.
    """
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=EXPECTED_SAMPLE_RATE)

    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert to numpy float32
    waveform_np = waveform.numpy().astype(np.float32)
    return waveform_np, EXPECTED_SAMPLE_RATE

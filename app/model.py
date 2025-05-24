# app/model.py

"""
Module to load the ONNX ASR model and run inference on 16kHz preprocessed audio features.
"""

import onnxruntime as ort
import numpy as np
import torchaudio
from nemo.collections.asr.parts.utils.decoder_configs import GreedyCTCDecoder
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.speaker_utils import prepare_segment_metadata
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the exported ONNX model
MODEL_PATH = "models/asr_conformer_hi.onnx"

class ASRModel:
    def __init__(self, model_path=MODEL_PATH):
        logger.info("Initializing ASRModel with ONNXRuntime...")
        self.session = ort.InferenceSession(model_path)

        # Vocabulary used by the original NeMo model
        self.vocab = list("ँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ०१२३४५६७८९।॥")

        self.decoder = GreedyCTCDecoder(vocabulary=self.vocab)
        logger.info("ASRModel loaded and ready.")

    def infer(self, audio_tensor: np.ndarray, sample_rate: int) -> str:
        """
        Perform inference on a preprocessed waveform.
        Args:
            audio_tensor: (1, time) np.ndarray of float32 PCM audio
            sample_rate: Sample rate (must be 16000)
        Returns:
            str: transcribed text
        """
        if sample_rate != 16000:
            raise ValueError("Audio must be sampled at 16kHz")

        # Feature extraction: using log mel filterbank features (as NeMo model expects)
        logger.info("Extracting features from audio...")

        audio_segment = AudioSegment(samples=audio_tensor[0], sample_rate=sample_rate)
        metadata = prepare_segment_metadata(audio_segment, max_duration=None)

        featurizer = FilterbankFeatures(
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            window="hann",
            normalize="per_feature",
            n_fft=512,
            n_filt=64,
            dither=1e-5,
        )

        features = featurizer(audio_segment.samples, metadata=metadata)
        features_np = features.unsqueeze(0).cpu().numpy()  # shape: (1, feat_dim, time)

        logger.info("Running inference...")
        ort_inputs = {"input": features_np}
        ort_outs = self.session.run(None, ort_inputs)
        logits = ort_outs[0]

        logger.info("Decoding transcription...")
        predictions = np.argmax(logits, axis=-1)
        transcript = self.decoder.decode(predictions)[0]

        return transcript.strip()

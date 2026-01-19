"""
MedASR Service - Medical Speech Recognition

This service uses Google's MedASR model to transcribe medical audio
with high accuracy for medical terminology.
"""

import torch
import librosa
from transformers import AutoModelForCTC, AutoProcessor, pipeline
from typing import Union, BinaryIO
import numpy as np
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class MedASRService:
    """Service for medical speech-to-text using MedASR."""
    
    def __init__(self):
        """Initialize MedASR model and processor."""
        self.device = settings.device
        self.model = None
        self.processor = None
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """Load MedASR model from Hugging Face."""
        try:
            logger.info(f"Loading MedASR model on device: {self.device}")
            
            # Use pipeline for easier inference
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=settings.medasr_model,
                device=0 if self.device == "cuda" else -1,
                token=settings.hf_token if settings.hf_token else None
            )
            
            logger.info("MedASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MedASR model: {e}")
            raise
    
    def transcribe(
        self, 
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = None
    ) -> str:
        """
        Transcribe audio to text using MedASR.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, M4A)
            audio_array: Audio array (if already loaded)
            sample_rate: Sample rate of audio_array
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio if path provided
            if audio_path:
                logger.info(f"Loading audio from: {audio_path}")
                audio_array, sample_rate = librosa.load(
                    audio_path, 
                    sr=settings.audio_sample_rate
                )
            elif audio_array is not None:
                # Resample if needed
                if sample_rate != settings.audio_sample_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sample_rate,
                        target_sr=settings.audio_sample_rate
                    )
                    sample_rate = settings.audio_sample_rate
            else:
                raise ValueError("Either audio_path or audio_array must be provided")
            
            # Check duration
            duration = len(audio_array) / sample_rate
            if duration > settings.max_audio_duration_seconds:
                raise ValueError(
                    f"Audio duration ({duration:.1f}s) exceeds maximum "
                    f"({settings.max_audio_duration_seconds}s)"
                )
            
            logger.info(f"Transcribing audio ({duration:.1f}s)...")
            
            # Transcribe using pipeline with chunking for long audio
            result = self.pipe(
                audio_array,
                chunk_length_s=20,  # Process in 20-second chunks
                stride_length_s=2   # 2-second overlap between chunks
            )
            
            transcript = result["text"]
            logger.info(f"Transcription complete: {len(transcript)} characters")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.pipe is not None


# Global instance (singleton pattern)
_medasr_service = None


def get_medasr_service() -> MedASRService:
    """Get or create MedASR service instance."""
    global _medasr_service
    if _medasr_service is None:
        _medasr_service = MedASRService()
    return _medasr_service

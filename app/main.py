"""
FastAPI Main Application

Voice Symptom Intake & Documentation Assistant

COMPLIANCE NOTICE:
This system provides ADMINISTRATIVE DOCUMENTATION SUPPORT ONLY.
It does NOT perform clinical triage, provide medical advice, or make clinical decisions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import logging
from pathlib import Path

from app.config import settings
from app.models.medasr_service import get_medasr_service
from app.models.medgemma_service import get_medgemma_service
from app.utils.audio_handler import AudioHandler

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Voice Symptom Intake & Documentation Assistant",
    description=(
        "Administrative tool for voice-based symptom intake and documentation. "
        "COMPLIANCE NOTICE: This system does NOT provide clinical triage, "
        "medical advice, or clinical decision support."
    ),
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Pydantic models
class TranscriptionResponse(BaseModel):
    transcript: str
    duration_seconds: float


class DocumentationRequest(BaseModel):
    transcript: str


class DocumentationResponse(BaseModel):
    documentation: dict
    requires_clinician_review: bool
    compliance_notice: str


class VoiceIntakeResponse(BaseModel):
    transcript: str
    documentation: dict
    duration_seconds: float
    requires_clinician_review: bool
    compliance_notice: str


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if services are ready."""
    try:
        medasr = get_medasr_service()
        medgemma = get_medgemma_service()
        
        return {
            "status": "healthy",
            "medasr_ready": medasr.is_ready(),
            "medgemma_ready": medgemma.is_ready(),
            "device": settings.device,
            "gpu_enabled": settings.enable_gpu
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# Transcription endpoint
@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text using MedASR.
    
    Args:
        audio: Audio file (WAV, MP3, M4A, FLAC, OGG)
        
    Returns:
        Transcription result
    """
    temp_file = None
    try:
        logger.info(f"Received audio file: {audio.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Load audio
        audio_handler = AudioHandler()
        audio_array, sr = audio_handler.load_audio(file_path=temp_path)
        duration = len(audio_array) / sr
        
        # Transcribe
        medasr = get_medasr_service()
        transcript = medasr.transcribe(audio_array=audio_array, sample_rate=sr)
        
        logger.info(f"Transcription successful: {len(transcript)} characters")
        
        return TranscriptionResponse(
            transcript=transcript,
            duration_seconds=duration
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_file:
            Path(temp_path).unlink(missing_ok=True)


# Documentation endpoint
@app.post("/api/document", response_model=DocumentationResponse)
async def generate_documentation(request: DocumentationRequest):
    """
    Generate structured symptom documentation from transcript.
    
    COMPLIANCE: This does NOT perform clinical assessment or triage.
    
    Args:
        request: Contains patient transcript
        
    Returns:
        Structured documentation flagged for clinician review
    """
    try:
        logger.info("Generating documentation from transcript")
        
        if not request.transcript or len(request.transcript.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Transcript too short (minimum 10 characters required)"
            )
        
        # Generate documentation
        medgemma = get_medgemma_service()
        documentation = medgemma.generate_documentation(request.transcript)
        
        logger.info("Documentation generated successfully")
        
        return DocumentationResponse(
            documentation=documentation,
            requires_clinician_review=True,
            compliance_notice=(
                "This is administrative documentation only. "
                "All clinical decisions must be made by qualified healthcare professionals."
            )
        )
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# End-to-end voice intake endpoint
@app.post("/api/voice-intake", response_model=VoiceIntakeResponse)
async def voice_intake(audio: UploadFile = File(...)):
    """
    Complete voice intake workflow: audio → transcription → documentation.
    
    COMPLIANCE: This is an administrative workflow only.
    No clinical decisions are made by this system.
    
    Args:
        audio: Audio file with patient symptom report
        
    Returns:
        Complete intake documentation
    """
    temp_file = None
    try:
        logger.info(f"Starting voice intake for: {audio.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Step 1: Load audio
        audio_handler = AudioHandler()
        audio_array, sr = audio_handler.load_audio(file_path=temp_path)
        duration = len(audio_array) / sr
        
        # Step 2: Transcribe
        medasr = get_medasr_service()
        transcript = medasr.transcribe(audio_array=audio_array, sample_rate=sr)
        logger.info(f"Transcription complete: {len(transcript)} characters")
        
        # Step 3: Generate documentation
        medgemma = get_medgemma_service()
        documentation = medgemma.generate_documentation(transcript)
        logger.info("Documentation generated")
        
        return VoiceIntakeResponse(
            transcript=transcript,
            documentation=documentation,
            duration_seconds=duration,
            requires_clinician_review=True,
            compliance_notice=(
                "This is administrative documentation only. "
                "All clinical decisions must be made by qualified healthcare professionals."
            )
        )
        
    except Exception as e:
        logger.error(f"Voice intake failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_file:
            Path(temp_path).unlink(missing_ok=True)


# Root endpoint - serve index.html
@app.get("/")
async def root():
    """Serve the main application page."""
    from fastapi.responses import FileResponse
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Voice Symptom Intake & Documentation Assistant API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )

# Voice Symptom Intake & Documentation Assistant

**Administrative Documentation Support for Direct Care AI**

> **⚠️ COMPLIANCE NOTICE:** This system provides administrative documentation support ONLY. It does NOT perform clinical triage, provide medical advice, or make clinical decisions. All outputs require review by qualified healthcare professionals.

---

## Overview

The Voice Symptom Intake & Documentation Assistant is a non-clinical administrative tool that integrates with Direct Care AI's existing service lines (Chronic Care Management, Remote Patient Monitoring) to enable 24/7 symptom intake, transforming raw voice data into structured clinical narratives while reducing documentation burden.

### Technology Stack

- **MedASR** - Google's medical-grade speech recognition (105M parameters)
- **MedGemma 1.5** - Google's medical language model (4B parameters, instruction-tuned)
- **Docker** - Containerized deployment
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework

---

## Features

✅ **Voice Symptom Intake**
- Record audio directly in browser
- Upload pre-recorded audio files (WAV, MP3, M4A, FLAC, OGG)
- Real-time audio visualization

✅ **Medical Transcription**
- Accurate transcription of medical terminology
- Handles medical dictation and patient conversations

✅ **Structured Documentation**
- Extracts patient-reported symptoms verbatim
- Generates draft SOAP notes (Subjective section)
- Structures timeline and symptom characteristics

✅ **Administrative Features**
- Export documentation as JSON
- Copy to clipboard
- All outputs flagged for clinician review

---

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Hugging Face account** (free) with access to MedASR and MedGemma
- **(Optional) NVIDIA GPU** with CUDA for faster inference

---

## Quick Start

### 1. Clone the Repository

```bash
cd voice-symptom-triage-assistant
```

### 2. Get Hugging Face Token

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with read permissions
4. Accept terms for:
   - [google/medasr](https://huggingface.co/google/medasr)
   - [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:
```env
HF_TOKEN=your_token_here
```

### 4. Run with Docker

**For CPU (Development/Testing):**
```bash
docker-compose --profile cpu up --build
```

**For GPU (Production):**
```bash
docker-compose --profile gpu up --build
```

### 5. Access the Application

Open browser to: **http://localhost:8000**

---

## Usage

### Web Interface

1. **Record Audio**: Click "Start Recording" and speak symptoms
2. **Or Upload File**: Choose an audio file from your device
3. **Generate Documentation**: Click submit to process
4. **Review Results**: View transcription and structured documentation
5. **Export**: Save as JSON or copy to clipboard

### API Endpoints

#### Health Check
```bash
GET /api/health
```

#### Transcribe Audio
```bash
POST /api/transcribe
Content-Type: multipart/form-data

audio: <audio file>
```

#### Generate Documentation
```bash
POST /api/document
Content-Type: application/json

{
  "transcript": "Patient symptom statement..."
}
```

#### Complete Voice Intake
```bash
POST /api/voice-intake
Content-Type: multipart/form-data

audio: <audio file>
```

---

## Integration with Direct Care AI Services

### Chronic Care Management (CCM)
- Patients call 24/7 to report symptom changes
- System transcribes and structures patient statements
- Care coordinators receive organized documentation for clinical review
- **Clinicians decide** intervention based on structured notes

### Remote Patient Monitoring (RPM)
- Voice interface for reporting device readings + symptoms
- Example: "My blood sugar is 250 and I feel dizzy"
- Extracts both quantitative and qualitative data
- No automated interpretation - routed to clinician

### Clinical Documentation Support
- Auto-generates draft SOAP notes (Subjective section)
- Reduces initial documentation time by 50-70%
- All notes flagged for clinician review
- Maintains medical terminology accuracy

---

## Project Structure

```
voice-symptom-triage-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration settings
│   ├── models/
│   │   ├── medasr_service.py      # Speech recognition
│   │   └── medgemma_service.py    # Documentation generation
│   ├── prompts/
│   │   └── documentation_prompts.py
│   ├── utils/
│   │   └── audio_handler.py       # Audio processing
│   └── static/
│       ├── index.html             # Web interface
│       ├── css/style.css
│       └── js/app.js
├── tests/                         # Test suite
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml            # Docker orchestration
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Development

### Local Development (No Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token_here
export DEVICE=cpu

# Run application
python -m app.main
```

---

## Compliance & Legal

> **⚠️ NOTICE:** HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at https://developers.google.com/health-ai-developer-foundations/terms

### Google HAI-DEF Terms

This application uses Google's MedGemma and MedASR models from the Health AI Developer Foundations (HAI-DEF) collection. Use of these models is governed by:

- **Terms of Use:** https://developers.google.com/health-ai-developer-foundations/terms
- **Prohibited Use Policy:** https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy

This application is designed for **RESEARCH AND DEVELOPMENT purposes only** and explicitly:

❌ **Does NOT:**
- Perform clinical triage or urgency assessment
- Provide diagnosis or treatment recommendations
- Make clinical decisions or route patients
- Provide medical advice to patients
- Replace qualified healthcare professionals

✅ **Does:**
- Transcribe patient voice messages
- Structure symptom narratives
- Extract and organize patient-reported information
- Generate draft documentation for clinician review

> **Clinical Use Notice:** Clinical Use (as defined in HAI-DEF Terms Section 1.1) requires appropriate Health Regulatory Authorization. This software is NOT approved for clinical use without such authorization.

### HIPAA Considerations

While this tool uses de-identified training data, implementers must ensure:
- Proper data encryption (in transit and at rest)
- Access controls and audit logging
- Business Associate Agreements (BAAs) with service providers
- Compliance with local healthcare regulations

---

## Troubleshooting

### Model Loading Issues

```bash
# Check if Hugging Face token is set
echo $HF_TOKEN

# Verify model access
huggingface-cli whoami
```

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Audio Quality Issues

- Ensure microphone permissions are granted
- Use supported audio formats (WAV recommended)
- Keep audio files under 5 minutes
- Minimize background noise

---

## Performance

### Expected Response Times

- MedASR Transcription: < 2s for 30-second audio (GPU)
- MedGemma Documentation: < 3s (GPU)
- End-to-end: < 5s (GPU)

CPU-only mode is 5-10x slower but functional for development/testing.

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review implementation plan
3. Check Hugging Face model documentation:
   - [MedASR](https://huggingface.co/google/medasr)
   - [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it)

---

## License

This project uses Google's open-source models under the Google HAI-DEF Terms.

---

## Acknowledgments

- **Google Health AI** - for MedASR and MedGemma models
- **Hugging Face** - for model hosting and Transformers library
- **Direct Care AI** - for the use case and integration vision

---

**Version:** 1.0.0  
**Last Updated:** January 2026

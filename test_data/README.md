# Sample Test Audio Instructions

This directory should contain test audio files for verification.

## Creating Test Audio

You can create test audio files by:

1. **Recording on your device:**
   - Use Windows Voice Recorder or similar
   - Record sample symptom statements
   - Save as WAV or MP3

2. **Example symptom statements:**
   - "I've had a headache and nausea for the past three hours"
   - "My blood sugar is 250 and I'm feeling dizzy"
   - "I have chest tightness that started last night"

3. **File format:**
   - Preferred: WAV, 16kHz, mono
   - Also supported: MP3, M4A, FLAC, OGG
   - Keep duration under 5 minutes

## Using Test Data

Place audio files here and reference them in tests:

```python
test_audio_path = "test_data/sample_symptom.wav"
```

## Notes

- Use clear audio with minimal background noise
- Speak naturally at normal pace
- Include medical terminology to test transcription accuracy

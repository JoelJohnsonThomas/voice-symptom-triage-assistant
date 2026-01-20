# MedGemma JSON Parsing Fixes - Testing Guide

## Quick Start

Run the integration test to validate the fixes:

```bash
cd e:\CSE\External\AI_Projects\voice-symptom-triage-assistant
python test_medgemma_integration.py
```

This will:
- Load the MedGemma model
- Process the sample transcript: "I had headache and nausea for the past three hours"
- Validate JSON parsing and field population
- Display detailed results

## What Was Fixed

### 1. Enhanced Prompt Engineering
- Added explicit JSON formatting instructions with markdown code fence
- Included concrete example showing expected output format
- Stronger directives: "RESPOND ONLY WITH THE JSON OBJECT"
- Instructed model to start response with `{` character

### 2. Robust JSON Extraction
The service now handles multiple response formats:
- Valid JSON directly in response
- JSON wrapped in markdown code fences (```json...```)
- JSON mixed with surrounding text
- Prompt echo that includes original instructions

### 3. Intelligent Fallback Parsing
When JSON parsing fails, the system:
- Uses regex to extract key fields from unstructured text
- Extracts chief complaint, symptoms, and SOAP notes
- Returns structured data even from conversational responses
- Flags the parsing method for transparency

### 4. Improved Generation Parameters
Added to `config.py`:
- `temperature=0.1` - More deterministic, structured output
- `max_tokens=1024` - Ensures complete documentation
- `repetition_penalty=1.1` - Prevents repetitive text

### 5. Better Logging
- Logs raw model output (first 200 chars)
- Logs extracted JSON text before parsing
- Indicates which parsing method succeeded
- More detailed error messages

## Expected Results

### Success Case (JSON Parsing Works)
```json
{
  "chief_complaint": "headache and nausea",
  "symptom_details": {
    "symptoms_mentioned": ["headache", "nausea"],
    "onset": "3 hours ago",
    "duration": "3 hours",
    ...
  },
  "soap_note_subjective": "Patient reports experiencing...",
  "parsing_method": "json_successful",
  "requires_clinician_review": true
}
```

### Fallback Case (Text Extraction Used)
```json
{
  "chief_complaint": "I had headache and nausea for the past three hours",
  "symptom_details": {
    "symptoms_mentioned": ["headache", "nausea"],
    ...
  },
  "soap_note_subjective": "Patient reports: I had headache...",
  "parsing_method": "text_extraction_fallback",
  "raw_text": "...",
  "requires_clinician_review": true
}
```

## Validation Checklist

When testing, verify:
- ✓ Chief Complaint is populated (not "N/A")
- ✓ SOAP Note Subjective is populated (not "N/A")
- ✓ Symptoms are extracted
- ✓ `parsing_method` field indicates success or fallback
- ✓ `requires_clinician_review` is set to `true`
- ✓ No urgency/severity fields present (compliance)

## Troubleshooting

### If JSON Parsing Still Fails
Check the logs for:
1. **Raw model output** - Is the model generating JSON at all?
2. **Extracted JSON text** - Is the extraction logic finding the JSON?
3. **Parsing method** - Which strategy worked?

### If Fields Are Still "N/A"
- The fallback extraction should prevent this
- Check `raw_text` field to see what the model actually generated
- May indicate model needs more specific prompting for your use case

### Model Loading Issues
If the model fails to load:
- Verify Hugging Face token is set in `.env`
- Check available memory (model requires ~5GB)
- Ensure internet connection for initial download

## Testing on Google Colab

If deploying to Colab, the test script works there too:

```python
# In Colab
!cd /content/voice-symptom-triage-assistant && python test_medgemma_integration.py
```

## Next Steps

1. Run the integration test locally
2. If successful, test with the full application:
   - Start the server: `python -m uvicorn main:app --reload`
   - Upload test audio or record
   - Verify documentation displays correctly
3. Test with various symptom descriptions
4. Monitor logs for any parsing method switches

## Files Modified

- `app/prompts/documentation_prompts.py` - Enhanced prompt
- `app/models/medgemma_service.py` - Robust JSON extraction + fallback
- `app/config.py` - Generation parameters
- `test_medgemma_integration.py` - New test script (NEW)

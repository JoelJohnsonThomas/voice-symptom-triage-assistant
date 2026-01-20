"""
Compliant Documentation Prompts for MedGemma

COMPLIANCE NOTICE:
These prompts are designed to extract and structure information ONLY.
They explicitly prohibit clinical decision-making, triage, or urgency assessment.
"""

SYMPTOM_DOCUMENTATION_PROMPT = """Extract symptom information from this patient statement and return ONLY valid JSON.

Patient: {transcript}

Return this exact JSON structure (replace with actual extracted info):
{{
  "chief_complaint": "main symptom",
  "symptom_details": {{
    "symptoms_mentioned": ["symptom1", "symptom2"],
    "onset": "when started",
    "duration": "how long",
    "location": "where",
    "quality": "description",
    "severity_description": "patient's words",
    "associated_symptoms": ["other symptoms"],
    "aggravating_factors": "what worsens",
    "alleviating_factors": "what helps"
  }},
  "soap_note_subjective": "Detailed SOAP note text"
}}

Rules:
- Use "not specified" for missing info
- Extract ONLY what patient said
- No medical advice
- No triage/urgency assessment

Output (start immediately with {{):"""


def create_documentation_prompt(transcript: str) -> str:
    """
    Create a compliant documentation prompt for MedGemma.
    
    Args:
        transcript: Patient's symptom report
        
    Returns:
        Formatted prompt string
    """
    return SYMPTOM_DOCUMENTATION_PROMPT.format(transcript=transcript)

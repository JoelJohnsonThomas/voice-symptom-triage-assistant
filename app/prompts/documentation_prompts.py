"""
Compliant Documentation Prompts for MedGemma

COMPLIANCE NOTICE:
These prompts are designed to extract and structure information ONLY.
They explicitly prohibit clinical decision-making, triage, or urgency assessment.
"""

def create_documentation_prompt(transcript: str) -> str:
    """
    Create clean, direct prompt without chat artifacts.
    Cleans transcript before formatting to remove ASR artifacts.
    
    Args:
        transcript: Patient's symptom report
        
    Returns:
        Formatted prompt string
    """
    # Clean the transcript first - remove ASR special tokens
    clean_transcript = transcript.replace("</s>", "").replace("<s>", "").strip().lstrip('.')
    
    # Direct instruction asking for structured extraction + narrative SOAP
    return f"""Analyze this patient statement for medical documentation.

Patient Statement: "{clean_transcript}"

Extract ONLY information the patient explicitly stated:
1. Main symptoms (comma-separated list)
2. Duration/timing (when it started or how long)
3. Brief SOAP Subjective note (1-2 sentences summarizing the history of present illness)

Do not add symptoms not stated. Use plain English, no markdown."""


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
2. Location (body part affected, include any radiation pattern like "back, radiating to leg")
3. Quality/Character (how the symptom feels: sharp, dull, burning, throbbing, aching, etc.)
4. Duration/timing (when it started or how long, e.g. "2 days", "since Monday", "chronic")
5. Severity (if patient describes intensity: mild, moderate, severe, or numeric scale)
6. Associated symptoms (other symptoms mentioned alongside the main complaint)
7. Brief SOAP Subjective note (1-2 sentences summarizing the history of present illness, include radiation patterns if mentioned)

Important extraction rules:
- For radiation patterns (e.g. "radiating to", "spreading to", "goes down to"), include in Location field
- If patient says "back pain radiating to leg", Location should be "back, radiating to leg"
- Do not add symptoms or details not stated by patient
- Use "not specified" for fields without explicit information
- Use plain English, no markdown formatting."""


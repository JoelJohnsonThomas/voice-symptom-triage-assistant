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
    
    # Simple, direct instruction with no chat template tokens
    return f"""Analyze this patient statement for medical documentation.

Patient Statement: "{clean_transcript}"

Extract ONLY information explicitly mentioned:
- Main symptoms (list only symptoms the patient stated)
- Duration or onset (if mentioned)
- Associated details (if mentioned)

Do not add symptoms not stated. Use plain English, no markdown formatting."""

"""
Compliant Documentation Prompts for MedGemma

COMPLIANCE NOTICE:
These prompts are designed to extract and structure information ONLY.
They explicitly prohibit clinical decision-making, triage, or urgency assessment.
"""

SYMPTOM_DOCUMENTATION_PROMPT = """Based on this patient statement, please provide the symptom information:

Statement: "{transcript}"

Please provide:
1. What is the main symptom or complaint?
2. When did it start and how long has it lasted?
3. Any other relevant details the patient mentioned?

Answer concisely."""


def create_documentation_prompt(transcript: str) -> str:
    """
    Create a compliant documentation prompt for MedGemma.
    
    Args:
        transcript: Patient's symptom report
        
    Returns:
        Formatted prompt string
    """
    return SYMPTOM_DOCUMENTATION_PROMPT.format(transcript=transcript)

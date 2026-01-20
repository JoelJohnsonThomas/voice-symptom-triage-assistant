"""
MedGemma Service - Medical Documentation Generation

COMPLIANCE NOTICE:
This service generates ADMINISTRATIVE DOCUMENTATION ONLY.
It does NOT provide clinical triage, urgency assessment, or medical advice.
All outputs are flagged for mandatory clinician review.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class MedGemmaService:
    """Service for medical documentation generation using MedGemma 1.5."""
    
    def __init__(self):
        """Initialize MedGemma model and processor."""
        self.device = settings.device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load MedGemma model from Hugging Face."""
        try:
            logger.info(f"Loading MedGemma model on device: {self.device}")
            
            # Use bfloat16 on GPU for better numerical stability (recommended for MedGemma)
            # float16 can cause empty output issues with this model
            if settings.enable_gpu and torch.cuda.is_available():
                dtype = torch.bfloat16
                logger.info("Using bfloat16 precision on GPU for numerical stability")
            else:
                # Use float32 on CPU
                dtype = torch.float32
                logger.info("Using float32 precision on CPU")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.medgemma_model,
                token=settings.hf_token if settings.hf_token else None
            )
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.medgemma_model,
                torch_dtype=dtype,
                device_map="auto" if settings.enable_gpu and torch.cuda.is_available() else None,
                token=settings.hf_token if settings.hf_token else None,
                low_cpu_mem_usage=True
            )
            
            # Manual device placement if not using device_map
            if not (settings.enable_gpu and torch.cuda.is_available()):
                self.model = self.model.to("cpu")
                self.device = "cpu"
                logger.info("MedGemma running on CPU (GPU disabled or unavailable)")
            else:
                logger.info(f"MedGemma running on GPU with device_map=auto")
            
            self.model.eval()
            
            logger.info(f"MedGemma model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma model: {e}")
            raise
    
    def _extract_json_from_response(self, text: str) -> str:
        """
        Extract JSON from model response, handling various formats.
        
        Args:
            text: Raw model response
            
        Returns:
            Extracted JSON string
        """
        import re
        
        # Remove prompt echo if present (common with instruction models)
        # The model often echoes the entire prompt before generating
        if "RESPOND ONLY WITH THE JSON OBJECT" in text:
            # Split after this marker and take everything after
            parts = text.split("RESPOND ONLY WITH THE JSON OBJECT")
            if len(parts) > 1:
                # Take the response part (after the prompt)
                text = parts[-1]
                logger.info(f"Removed prompt echo, remaining text length: {len(text)}")
        
        # Try to extract JSON from markdown code fence
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            logger.info("Found JSON in markdown code fence")
            return json_match.group(1).strip()
        
        # Try to find JSON object using non-greedy matching
        # Look for { followed by anything (non-greedy) followed by }
        # This should match the first complete JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            logger.info("Found JSON using non-greedy pattern matching")
            return json_match.group(0).strip()
        
        # Last resort: try greedy matching from first { to last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            logger.info(f"Found JSON using brace positions: {first_brace} to {last_brace}")
            return json_str.strip()
        
        logger.warning("No JSON pattern found in response")
        # Return as-is if no pattern found
        return text.strip()
    
    def _extract_fields_from_text(self, text: str, transcript: str) -> Dict[str, Any]:
        """
        Fallback: Extract fields from conversational text when JSON parsing fails.
        
        Args:
            text: Raw model response (conversational)
            transcript: Original patient transcript
            
        Returns:
            Dictionary with extracted documentation fields
        """
        import re
        
        # Clean up the text
        text = text.strip()
        
        # Common symptom keywords to look for
        symptom_keywords = [
            "headache", "migraine", "pain", "ache",
            "nausea", "vomiting", "sick",
            "fever", "temperature", "hot", "chills",
            "dizziness", "dizzy", "lightheaded",
            "cough", "congestion", "cold",
            "fatigue", "tired", "weak",
            "rash", "itching", "swelling"
        ]
        
        # Extract chief complaint from transcript or model output
        chief_complaint = "not specified"
        symptoms_found = []
        
        # Check both transcript and model response for symptoms
        combined_text = (transcript + " " + text).lower()
        for keyword in symptom_keywords:
            if keyword in combined_text:
                symptoms_found.append(keyword)
        
        if symptoms_found:
            chief_complaint = ", ".join(symptoms_found[:3])  # Top 3 symptoms
        else:
            # Use first meaningful phrase from transcript
            words = transcript.strip().split()
            if len(words) > 0:
                chief_complaint = " ".join(words[:5])  # First 5 words
        
        # Extract timing information
        onset = "not specified"
        duration = "not specified"
        
        # Look for time-related patterns
        time_patterns = [
            (r'(\d+)\s*(?:hours?|hrs?)', 'hours'),
            (r'(\d+)\s*(?:days?)', 'days'),
            (r'(\d+)\s*(?:weeks?)', 'weeks'),
            (r'past\s+(\d+\s+\w+)', 'duration')
        ]
        
        for pattern, time_type in time_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                if time_type == 'duration':
                    duration = match.group(1)
                    onset = match.group(1) + " ago"
                else:
                    duration = match.group(1) + " " + time_type
                    onset = duration + " ago"
                break
        
        # Create SOAP note
        soap_note = f"Patient reports {chief_complaint}"
        if duration != "not specified":
            soap_note += f" for {duration}"
        soap_note += f". {text[:200]}" if text else "."
        
        return {
            "chief_complaint": chief_complaint,
            "symptom_details": {
                "symptoms_mentioned": symptoms_found if symptoms_found else ["not specified"],
                "onset": onset,
                "duration": duration,
                "location": "not specified",
                "quality": "not specified",
                "severity_description": "not specified",
                "associated_symptoms": symptoms_found if symptoms_found else ["not specified"],
                "aggravating_factors": "not specified",
                "alleviating_factors": "not specified"
            },
            "soap_note_subjective": soap_note.strip(),
            "parsing_method": "text_extraction"
        }
    
    def generate_documentation(self, transcript: str) -> Dict[str, Any]:
        """
        Generate structured symptom documentation from transcript.
        
        COMPLIANCE: This does NOT perform triage or clinical assessment.
        It only extracts and structures information from the patient's statement.
        
        Args:
            transcript: Patient's symptom report (text)
            
        Returns:
            Dictionary with structured documentation
        """
        try:
            logger.info("Generating documentation...")
            
            # Import prompt here to avoid circular dependency
            from app.prompts.documentation_prompts import create_documentation_prompt
            
            # Create prompt content
            prompt_content = create_documentation_prompt(transcript)
            
            # MedGemma 1.5 is a chat model - use chat template
            messages = [
                {"role": "user", "content": prompt_content}
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.info(f"Formatted prompt (first 200 chars): {prompt[:200]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # Generate with greedy decoding (stable on GPU with float16)
            # We use conversational output and extract data via text parsing
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Shorter for focused answers
                    do_sample=False,  # Greedy is stable
                    repetition_penalty=settings.medgemma_repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output (skip the input prompt)
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"Raw model output (length={len(decoded)}): {decoded[:200]}...")
            
            # Extract documentation from conversational text
            documentation = self._extract_fields_from_text(decoded, transcript)
            
            # Ensure compliance fields are present
            documentation["requires_clinician_review"] = True
            documentation["compliance_notice"] = (
                "This is administrative documentation only. "
                "All clinical decisions must be made by qualified healthcare professionals."
            )
            
            # Remove any urgency/severity fields if present (compliance)
            documentation.pop("urgency", None)
            documentation.pop("severity", None)
            documentation.pop("risk_level", None)
            documentation.pop("recommended_actions", None)
            
            return documentation
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None


# Global instance (singleton pattern)
_medgemma_service = None


def get_medgemma_service() -> MedGemmaService:
    """Get or create MedGemma service instance."""
    global _medgemma_service
    if _medgemma_service is None:
        _medgemma_service = MedGemmaService()
    return _medgemma_service

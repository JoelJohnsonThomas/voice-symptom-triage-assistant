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
        Extract fields from conversational text with markdown cleanup.
        
        Args:
            text: Raw model response (may contain markdown)
            transcript: Original patient transcript
            
        Returns:
            Dictionary with extracted documentation fields
        """
        import re
        
        # Clean markdown formatting from model output
        cleaned_text = text
        cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)  # Remove **bold**
        cleaned_text = re.sub(r'_(.*?)_', r'\1', cleaned_text)        # Remove _italic_
        cleaned_text = re.sub(r'`(.*?)`', r'\1', cleaned_text)        # Remove `code`
        cleaned_text = re.sub(r'^\d+\.\s*', '', cleaned_text, flags=re.MULTILINE)  # Remove numbered lists
        
        # Common symptom keywords
        symptom_keywords = [
            "headache", "migraine", "pain", "ache",
            "nausea", "vomiting", "sick",
            "fever", "temperature", "hot", "chills",
            "dizziness", "dizzy", "lightheaded",
            "cough", "congestion", "cold",
            "fatigue", "tired", "weak",
            "rash", "itching", "swelling"
        ]
        
        # Extract chief complaint
        chief_complaint = "not specified"
        symptoms_found = []
        
        # Check both transcript and model response
        combined_text = (transcript + " " + cleaned_text).lower()
        for keyword in symptom_keywords:
            if keyword in combined_text:
                symptoms_found.append(keyword)
        
        if symptoms_found:
            chief_complaint = ", ".join(symptoms_found[:3])  # Top 3 symptoms
        else:
            # Try extracting from "Main Symptom:" pattern
            symptom_match = re.search(r'(?:main symptom|chief complaint)[:\s]*([^\.\n]+)', cleaned_text, re.IGNORECASE)
            if symptom_match:
                chief_complaint = symptom_match.group(1).strip()
            else:
                # Use first meaningful phrase from transcript
                words = transcript.strip().split()
                if len(words) > 0:
                    chief_complaint = " ".join(words[:5])
        
        # Extract timing information
        onset = "not specified"
        duration = "not specified"
        
        # Enhanced time patterns
        time_patterns = [
            (r'started\s+(\d+\s+(?:days?|hours?|weeks?))\s+ago', 'started'),
            (r'for\s+(?:the\s+)?past\s+(\d+\s+(?:days?|hours?|weeks?))', 'duration'),
            (r'(?:for|over|since)\s+(\d+\s+(?:days?|hours?|weeks?))', 'duration'),
            (r'(\d+)\s+(days?|hours?|weeks?)', 'duration')
        ]
        
        for pattern, match_type in time_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                time_value = match.group(1)
                if match_type == 'started':
                    onset = time_value
                    duration = time_value
                else:
                    duration = time_value
                    onset = duration + " ago" if "ago" not in duration else duration
                break
        
        # Build clean SOAP note from model output + extracted info
        soap_parts = []
        
        if symptoms_found:
            soap_parts.append(f"Patient reports {chief_complaint}")
        
        if duration != "not specified":
            soap_parts.append(f"symptoms present for {duration}")
        
        # Add relevant excerpts from model output (first clean sentence)
        sentences = [s.strip() for s in re.split(r'[\.!?]', cleaned_text) if s.strip()]
        if sentences and len(soap_parts) < 2:
            # Add first informative sentence from model
            for sent in sentences[:2]:
                if len(sent) > 10 and any(word in sent.lower() for word in ['symptom', 'patient', 'report']):
                    soap_parts.append(sent)
                    break
        
        soap_note = ". ".join(soap_parts).strip()
        if not soap_note.endswith('.'):
            soap_note += "."
        
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
            "soap_note_subjective": soap_note,
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
            
            # Generate with greedy decoding (stable on GPU with bfloat16)
            # We use conversational output and extract data via text parsing
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased to avoid truncation
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

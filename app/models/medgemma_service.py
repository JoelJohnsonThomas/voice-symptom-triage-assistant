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
            
            # Determine optimal dtype based on device
            if settings.enable_gpu and torch.cuda.is_available():
                # Use float16 on GPU to reduce VRAM usage (4.96GB -> ~2.5GB)
                dtype = torch.float16
                logger.info("Using float16 precision on GPU for memory efficiency")
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
        # Look for the actual JSON output after the prompt
        if "Patient Statement:" in text:
            # Split after the prompt and take the response part
            parts = text.split("RESPOND ONLY WITH THE JSON OBJECT")
            if len(parts) > 1:
                text = parts[-1]
        
        # Try to extract JSON from markdown code fence
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find JSON object directly (between curly braces)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        # Return as-is if no pattern found
        return text.strip()
    
    def _extract_fields_from_text(self, text: str, transcript: str) -> Dict[str, Any]:
        """
        Fallback: Extract fields from unstructured text when JSON parsing fails.
        
        Args:
            text: Raw model response
            transcript: Original patient transcript
            
        Returns:
            Dictionary with extracted documentation fields
        """
        import re
        
        # Try to extract chief complaint
        chief_complaint = "unknown"
        cc_match = re.search(r'"?chief_complaint"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
        if cc_match:
            chief_complaint = cc_match.group(1)
        else:
            # Extract first sentence from transcript as fallback
            sentences = transcript.split('.')
            if sentences:
                chief_complaint = sentences[0].strip()
        
        # Try to extract SOAP note
        soap_note = ""
        soap_match = re.search(r'"?soap_note_subjective"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
        if soap_match:
            soap_note = soap_match.group(1)
        else:
            # Generate basic SOAP note from transcript
            soap_note = f"Patient reports: {transcript}"
        
        # Try to extract symptoms
        symptoms = []
        symptoms_match = re.search(r'"?symptoms_mentioned"?\s*[:=]\s*\[(.*?)\]', text, re.DOTALL)
        if symptoms_match:
            symptoms_text = symptoms_match.group(1)
            symptoms = [s.strip().strip('"\'') for s in symptoms_text.split(',')]
        
        return {
            "chief_complaint": chief_complaint,
            "symptom_details": {
                "symptoms_mentioned": symptoms if symptoms else ["not specified"],
                "onset": "not specified",
                "duration": "not specified",
                "location": "not specified",
                "quality": "not specified",
                "severity_description": "not specified",
                "associated_symptoms": symptoms if symptoms else ["not specified"],
                "aggravating_factors": "not specified",
                "alleviating_factors": "not specified"
            },
            "soap_note_subjective": soap_note,
            "parsing_method": "text_extraction_fallback"
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
            
            # Create prompt
            prompt = create_documentation_prompt(transcript)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # Generate with improved parameters
            # Use greedy decoding (do_sample=False) for deterministic, stable output on GPU
            # Sampling with low temperature can cause numerical issues with float16
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.medgemma_max_tokens,
                    do_sample=False,  # Greedy decoding for stability on GPU
                    repetition_penalty=settings.medgemma_repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Raw model output (length={len(decoded)}): {decoded[:200]}...")
            
            # Extract JSON from response
            json_text = self._extract_json_from_response(decoded)
            
            logger.info(f"Extracted JSON text (length={len(json_text)}): {json_text[:200]}...")
            
            # Parse JSON response
            try:
                documentation = json.loads(json_text)
                logger.info("Successfully parsed JSON response")
                
                # Ensure compliance fields are present
                documentation["requires_clinician_review"] = True
                documentation["compliance_notice"] = (
                    "This is administrative documentation only. "
                    "All clinical decisions must be made by qualified healthcare professionals."
                )
                documentation["parsing_method"] = "json_successful"
                
                # Remove any urgency/severity fields if present (compliance)
                documentation.pop("urgency", None)
                documentation.pop("severity", None)
                documentation.pop("risk_level", None)
                documentation.pop("recommended_actions", None)
                
                return documentation
                
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse JSON response: {je}, attempting text extraction")
                
                # Fallback: Extract fields from unstructured text
                documentation = self._extract_fields_from_text(decoded, transcript)
                
                # Add compliance fields
                documentation["requires_clinician_review"] = True
                documentation["compliance_notice"] = (
                    "This is administrative documentation only. "
                    "All clinical decisions must be made by qualified healthcare professionals."
                )
                documentation["raw_text"] = decoded
                
                logger.info(f"Extracted documentation using fallback method: {list(documentation.keys())}")
                
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

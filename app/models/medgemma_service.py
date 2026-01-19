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
            
            # MedGemma 1.5 4B (4.96GB) exceeds GTX 1650's 4GB VRAM
            # Force CPU execution to avoid VRAM overflow
            logger.warning("MedGemma running on CPU due to VRAM constraints (4.96GB model > 4GB VRAM)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.medgemma_model,
                token=settings.hf_token if settings.hf_token else None
            )
            
            # Load with float32 on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.medgemma_model,
                torch_dtype=torch.float32,
                device_map=None,  # Disable auto device mapping
                token=settings.hf_token if settings.hf_token else None,
                low_cpu_mem_usage=True  # Optimize CPU memory usage
            )
            
            # Force model to CPU
            self.model = self.model.to("cpu")
            self.device = "cpu"
            
            self.model.eval()
            
            logger.info("MedGemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma model: {e}")
            raise
    
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
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info("Documentation generated successfully")
            
            # Parse JSON response
            try:
                documentation = json.loads(decoded)
                
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
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                return {
                    "raw_documentation": decoded,
                    "requires_clinician_review": True,
                    "compliance_notice": (
                        "This is administrative documentation only. "
                        "All clinical decisions must be made by qualified healthcare professionals."
                    )
                }
            
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

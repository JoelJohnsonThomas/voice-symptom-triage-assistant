"""
Test script to validate MedGemma JSON parsing improvements.

This script tests the MedGemma service with a sample transcription
to verify that JSON parsing works correctly and all fields are populated.
"""

import sys
import os
import json
import logging

# Add the app directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.models.medgemma_service import get_medgemma_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_medgemma_documentation():
    """Test MedGemma documentation generation with sample transcript."""
    
    print("\n" + "="*80)
    print("MEDGEMMA JSON PARSING TEST")
    print("="*80 + "\n")
    
    # Sample transcript (the one that was failing)
    transcript = "I had headache and nausea for the past three hours"
    
    print(f"Input Transcript: {transcript}\n")
    print("Loading MedGemma service...")
    
    try:
        # Get MedGemma service
        service = get_medgemma_service()
        
        if not service.is_ready():
            print("❌ ERROR: MedGemma service is not ready")
            return False
        
        print("✓ MedGemma service loaded successfully\n")
        print("Generating documentation...\n")
        
        # Generate documentation
        result = service.generate_documentation(transcript)
        
        print("="*80)
        print("RESULT")
        print("="*80 + "\n")
        
        # Pretty print the result
        print(json.dumps(result, indent=2))
        
        print("\n" + "="*80)
        print("VALIDATION")
        print("="*80 + "\n")
        
        # Validate the result
        validation_passed = True
        
        # Check for chief complaint
        if "chief_complaint" in result and result["chief_complaint"] not in ["N/A", "unknown", ""]:
            print(f"✓ Chief Complaint: {result['chief_complaint']}")
        else:
            print(f"❌ Chief Complaint is missing or N/A: {result.get('chief_complaint', 'MISSING')}")
            validation_passed = False
        
        # Check for SOAP note
        if "soap_note_subjective" in result and result["soap_note_subjective"] not in ["N/A", "", "unknown"]:
            print(f"✓ SOAP Note Subjective: {result['soap_note_subjective'][:100]}...")
        else:
            print(f"❌ SOAP Note is missing or N/A: {result.get('soap_note_subjective', 'MISSING')}")
            validation_passed = False
        
        # Check for symptom details
        if "symptom_details" in result:
            symptoms = result["symptom_details"].get("symptoms_mentioned", [])
            if symptoms and symptoms != ["not specified"]:
                print(f"✓ Symptoms Mentioned: {symptoms}")
            else:
                print(f"⚠ Symptoms are 'not specified' (may need manual review)")
        else:
            print(f"❌ Symptom details missing")
            validation_passed = False
        
        # Check parsing method
        parsing_method = result.get("parsing_method", "unknown")
        print(f"\nParsing Method: {parsing_method}")
        
        if parsing_method == "json_successful":
            print("✓ JSON parsing succeeded!")
        elif parsing_method == "text_extraction_fallback":
            print("⚠ JSON parsing failed, but text extraction fallback succeeded")
        else:
            print("⚠ Unknown parsing method")
        
        # Check compliance fields
        if result.get("requires_clinician_review") == True:
            print("✓ Compliance: Requires clinician review flag is set")
        else:
            print("❌ Compliance: Missing clinician review flag")
            validation_passed = False
        
        print("\n" + "="*80)
        
        if validation_passed:
            print("✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("⚠ SOME VALIDATION CHECKS FAILED (see above)")
        
        print("="*80 + "\n")
        
        return validation_passed
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nStarting MedGemma JSON Parsing Test...")
    print("This may take a few minutes as the model needs to load.\n")
    
    success = test_medgemma_documentation()
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠ Test completed with warnings or errors")
        sys.exit(1)

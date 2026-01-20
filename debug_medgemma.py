# MedGemma Debug Script for Google Colab
# This will capture and display MedGemma's actual output

print("Installing debug wrapper for MedGemma...")

# Read current medgemma_service.py
with open('/content/voice-symptom-triage-assistant/app/models/medgemma_service.py', 'r') as f:
    content = f.read()

# Add comprehensive logging right after decoding
debug_code = '''
            # ========== DEBUG LOGGING ==========
            import sys
            print("\\n" + "="*80, file=sys.stderr)
            print("MEDGEMMA DEBUG OUTPUT", file=sys.stderr)
            print("="*80, file=sys.stderr)
            print(f"PROMPT SENT:\\n{prompt[:500]}...\\n", file=sys.stderr)
            print("="*80, file=sys.stderr)
            print(f"RAW RESPONSE (length={len(decoded)}):\\n{decoded}\\n", file=sys.stderr)
            print("="*80 + "\\n", file=sys.stderr)
            
            # Also save to file for later inspection
            with open('/tmp/medgemma_debug.txt', 'w') as debug_f:
                debug_f.write(f"PROMPT:\\n{prompt}\\n\\n")
                debug_f.write(f"RAW OUTPUT:\\n{decoded}\\n")
            # ========== END DEBUG ==========
'''

# Find where to insert (after tokenizer.decode)
if 'decoded = self.tokenizer.decode' in content:
    content = content.replace(
        'decoded = self.tokenizer.decode(generation, skip_special_tokens=True)',
        'decoded = self.tokenizer.decode(generation, skip_special_tokens=True)' + debug_code
    )
    
    # Write back
    with open('/content/voice-symptom-triage-assistant/app/models/medgemma_service.py', 'w') as f:
        f.write(content)
    
    print("✅ Debug logging installed!")
    print("\nNext steps:")
    print("1. Restart your server")
    print("2. Record and submit audio")
    print("3. Look at the Colab output for the debug info between the === lines")
    print("4. OR run: !cat /tmp/medgemma_debug.txt")
else:
    print("❌ Could not find the decode line - file may have been modified")
    print("Please share the medgemma_service.py file")

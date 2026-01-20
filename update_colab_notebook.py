"""
Script to update Colab notebook to remove MEDGEMMA_TEMPERATURE parameter.
This prevents CUDA sampling errors on GPU by using greedy decoding instead.
"""

import json

# Read the notebook
with open('colab_deployment.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and update Step 4 cell (environment configuration)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and any('MEDGEMMA_TEMPERATURE' in line for line in cell.get('source', [])):
        # Update the source code
        source = cell['source']
        
        # Remove the temperature line
        source = [line for line in source if 'MEDGEMMA_TEMPERATURE' not in line]
        
        # Update the comment
        for i, line in enumerate(source):
            if 'optimized for JSON output' in line:
                source[i] = line.replace('optimized for JSON output', 'optimized for stable GPU inference')
        
        # Update the print statements
        for i, line in enumerate(source):
            if 'Environment configured with enhanced MedGemma parameters' in line:
                source[i] = 'print("✅ Environment configured with optimized MedGemma parameters!")\\n'
            elif 'Temperature: 0.1' in line:
                source[i] = 'print("   - Max Tokens: 1024 (complete documentation)")\\n'
            elif source[i].strip().startswith('print("   - Max Tokens:'):
                source[i] = 'print("   - Repetition Penalty: 1.1 (prevent loops)")\\n'
            elif source[i].strip().startswith('print("   - Repetition Penalty:'):
                source[i] = 'print("   - Using greedy decoding for stable GPU inference)")\\n'
        
        cell['source'] = source
        print(f"✓ Updated cell: Removed MEDGEMMA_TEMPERATURE parameter")

# Write back the updated notebook
with open('colab_deployment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print("\n✅ Colab notebook updated successfully!")
print("   Removed: MEDGEMMA_TEMPERATURE=0.1")
print("   Using: Greedy decoding for stable GPU inference")

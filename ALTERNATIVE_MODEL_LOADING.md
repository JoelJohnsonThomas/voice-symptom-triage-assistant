# Alternative MedGemma Loading Method

## If Chat Template Fix Doesn't Work

If the chat template fix doesn't resolve the empty output issue, try this alternative loading method:

### Changes to Make:

1. **Pin Transformers Version**:
```bash
pip install transformers==4.51.3
```

2. **Use bfloat16 instead of float16**:

In `app/models/medgemma_service.py`, line ~38:

```python
# Current:
dtype = torch.float16 if settings.enable_gpu and torch.cuda.is_available() else torch.float32

# Replace with:
dtype = torch.bfloat16 if settings.enable_gpu and torch.cuda.is_available() else torch.float32
```

3. **Optional: Use AutoProcessor** (for vision capabilities):

```python
# Current:
from transformers import AutoTokenizer, AutoModelForCausalLM

self.tokenizer = AutoTokenizer.from_pretrained(...)

# Replace with:
from transformers import AutoProcessor, AutoModelForCausalLM

self.tokenizer = AutoProcessor.from_pretrained(...)
```

**Note**: We can keep `AutoModelForCausalLM` for text-only usage. `AutoModelForImageTextToText` is only needed if you plan to add image inputs later.

## Why This Might Help

- **bfloat16**: Better numerical range, less overflow/underflow
- **Older Transformers**: Avoids known bugs in 4.52.3
- **AutoProcessor**: Handles both text and potential image inputs

## Test Order

1. ✅ **Try chat template fix first** (already implemented)
2. ❌ If still broken → Try bfloat16
3. ❌ If still broken → Downgrade transformers
4. ❌ If still broken → Switch to AutoProcessor

Don't change everything at once - debug incrementally!

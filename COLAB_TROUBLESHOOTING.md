# Google Colab Deployment Troubleshooting

## Error: 502/503 "Server responded with status 503" or "Unexpected token '<'"

### **Root Cause:**
The FastAPI server is crashing during startup, likely due to model loading failures.

### **Solution Steps:**

## 1. **Check Colab Server Logs**

In your Colab notebook, **look at the server output** for error messages. You should see:

```
INFO - Loading MedASR model on device: cuda
ERROR - Failed to load MedASR model: [ERROR MESSAGE HERE]
```

**Common Errors:**

### A. **Out of Memory (OOM)**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix:** Restart runtime and ensure no other models are loaded:
```python
# In Colab, before starting server
import torch
torch.cuda.empty_cache()
```

### B. **Hugging Face Token Error**
```
HTTPError: 401 Client Error: Unauthorized
```

**Fix:** 
1. Check your HF_TOKEN is correct
2. Ensure you accepted terms for both models:
   - https://huggingface.co/google/medasr
   - https://huggingface.co/google/medgemma-1.5-4b-it

### C. **Transformers Version Issue**
```
model type `lasr_ctc` not recognized
```

**Fix:** Ensure you installed transformers from the specific commit:
```python
!pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5
```

## 2. **Test Models Individually**

Add this cell **BEFORE** starting the server to test model loading:

```python
# Test MedASR loading
print("Testing MedASR...")
import torch
from transformers import pipeline

try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="google/medasr",
        device=0,
        token=HF_TOKEN
    )
    print("✅ MedASR loaded successfully!")
except Exception as e:
    print(f"❌ MedASR failed: {e}")

# Test MedGemma loading
print("\\nTesting MedGemma...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(
        "google/medgemma-1.5-4b-it",
        token=HF_TOKEN
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    print("✅ MedGemma loaded successfully!")
except Exception as e:
    print(f"❌ MedGemma failed: {e}")
```

## 3. **Disable Health Check Model Loading**

If models take too long to load, the health check times out. Update `app/main.py`:

```python
@app.get("/api/health")
async def health_check():
    """Quick health check without loading models"""
    return {
        "status": "healthy",
        "message": "Service is running"
    }
```

Then load models **lazily** on first request instead of at startup.

## 4. **Start Server with Debug Logging**

Change the server start command to show detailed errors:

```python
!python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## 5. **Use Colab's Built-in Server**

Instead of ngrok, try Colab's built-in public URLs:

```python
from google.colab import output
output.serve_kernel_port_as_window(8000)
```

Then run:
```python
!python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. **Simplified Startup (MedASR Only)**

If MedGemma is causing issues, temporarily disable it:

In `app/main.py`, comment out MedGemma service:

```python
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global medasr_service  # , medgemma_service
    
    try:
        # Load MedASR
        medasr_service = get_medasr_service()
        logger.info("MedASR service initialized")
        
        # # Disable MedGemma for testing
        # medgemma_service = get_medgemma_service()
        # logger.info("MedGemma service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
```

## 7. **Check Colab GPU Allocation**

Ensure you have GPU enabled:
- Runtime → Change runtime type
- Hardware accelerator → **GPU** (T4)
- Click **Save**

## Quick Fix: Updated Colab Cell

Replace Step 7 in the notebook with:

```python
import subprocess
import threading
import time

# Test server startup first (without ngrok)
print("Testing server startup...")
proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
time.sleep(30)  # Give models time to load

# Check if server is running
if proc.poll() is None:
    print("✅ Server started successfully!")
    
    # Now set up ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"\\n{'='*60}")
    print(f"🌐 PUBLIC URL: {public_url}")
    print(f"{'='*60}\\n")
    
    # Keep running
    proc.wait()
else:
    # Server crashed, show error
    stdout, stderr = proc.communicate()
    print("❌ Server failed to start!")
    print("\\nError output:")
    print(stderr.decode())
```

## Still Having Issues?

**Share the following information:**
1. Full error message from Colab server logs
2. Your GPU type (from `!nvidia-smi`)
3. Transformers version (from `!pip show transformers`)
4. Whether you accepted model terms on Hugging Face

This will help diagnose the exact issue!

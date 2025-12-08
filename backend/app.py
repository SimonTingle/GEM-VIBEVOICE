import os
import io
import logging
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts

# --- Conditional Imports for VibeVoice Dependencies ---
TORCH_AVAILABLE = False
try:
    import torch
    from accelerate import Accelerator
    import scipy.io.wavfile as wav
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch/Accelerate/Scipy not found. VibeVoice cannot be used.")

# --- CONFIGURATION ---
TRY_VIBEVOICE = True
# VibeVoice Sample Rate is 24 kHz
VIBEVOICE_SAMPLE_RATE = 24000 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceServer")

app = FastAPI()

# Allow Vercel to access this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADER ---
model = None
TTS_STRATEGY = "EdgeTTS (Cloud Fallback)"

if TRY_VIBEVOICE and TORCH_AVAILABLE:
    try:
        # Import the confirmed class for 0.5B Streaming
        from VibeVoice.vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference as VibeVoice
        
        logger.info("⏳ Loading VibeVoice Model (0.5B)...")
        
        accelerator = Accelerator()
        
        # Load and prepare model
        model = VibeVoice.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
        model = accelerator.prepare(model)
        model.eval()
        
        device = model.device if hasattr(model, 'device') else "CPU (Accelerated)"
        TTS_STRATEGY = f"VibeVoice (Local {device})"
        logger.info(f"✅ VibeVoice Loaded on {device}")
        
    except Exception as e:
        logger.warning(f"⚠️ VibeVoice failed to load: {e}")
        logger.warning("➡️ System will fallback to EdgeTTS (Cloud).")
        model = None
else:
    model = None

# --- AUDIO TENSOR EXTRACTION UTILITY ---
def extract_audio_tensor(output):
    """
    Robustly extracts the primary audio tensor from the complex VibeVoice model output.
    This resolves the persistent 'Attribute "audio" is unknown' errors.
    """
    if isinstance(output, torch.Tensor):
        return output
    
    # Try accessing common attributes of Hugging Face/VibeVoice model outputs
    if hasattr(output, 'audio'):
        return output.audio
    if hasattr(output, 'waveform'):
        return output.waveform
    if hasattr(output, 'sequences'):
        return output.sequences
    if isinstance(output, dict) and 'audio' in output:
        return output['audio']

    # Final attempt: If the output object is an iterable (like a tuple), try the first item
    try:
        if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]
    except:
        pass # Ignore errors if not iterable
        
    return None

# --- GENERATORS ---

async def generate_vibevoice_stream(text):
    """Generates audio using the local 0.5B model."""
    global model
    if model is None:
        return

    # Batch Generation (Most reliable path for this integration)
    try:
        with torch.no_grad():
            output = model.generate(text)
            
            # --- USE THE FIXED EXTRACTION UTILITY ---
            audio_tensor = extract_audio_tensor(output)

            # Validation and Conversion
            if audio_tensor is None or not hasattr(audio_tensor, 'cpu'):
                 logger.error(f"VibeVoice output error: Could not extract tensor. Final type: {type(output)}")
                 # Raise a 500 Internal Server Error to the client
                 raise HTTPException(status_code=500, detail="TTS generation failed: Model output format is incompatible.")
            
            # Convert to numpy array (moves to CPU and converts)
            # The squeeze() and numpy() calls are safe only after we ensure it's a tensor.
            audio_data = audio_tensor.cpu().numpy().squeeze()
            
            # Flatten to 1D array if needed (e.g., [1, N] -> [N])
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()

            # Write to a BytesIO object for in-memory WAV conversion
            byte_io = io.BytesIO()
            wav.write(byte_io, VIBEVOICE_SAMPLE_RATE, audio_data)
            
            yield byte_io.getvalue()
            
    except Exception as e:
        logger.error(f"VibeVoice runtime error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"VibeVoice internal failure: {e}")

async def generate_edgetts_stream(text):
    """Fallback: Uses Microsoft Edge Cloud TTS"""
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    async for chunk in communicate.stream():
        # This structure is safe and handles the TypedDict warning
        if chunk.get("type") == "audio" and chunk.get("data"):
            yield chunk["data"]

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "Online", "strategy": TTS_STRATEGY}

@app.post("/speak")
async def speak(payload: dict = Body(...)):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    logger.info(f"Speaking via {TTS_STRATEGY}: {text[:30]}...")

    if model:
        # If VibeVoice is loaded, use it (assumes WAV output)
        return StreamingResponse(
            generate_vibevoice_stream(text),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"}
        )
    else:
        # Use EdgeTTS fallback (assumes MPEG/MP3 output)
        return StreamingResponse(
            generate_edgetts_stream(text),
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache"}
        )

if __name__ == "__main__":
    if model and TORCH_AVAILABLE and not torch.cuda.is_available():
        logger.warning("Running on CPU. Expect high latency.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
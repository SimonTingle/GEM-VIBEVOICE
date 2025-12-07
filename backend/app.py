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
    # This warning is useful for local development without a full VibeVoice setup
    print("Warning: PyTorch/Accelerate/Scipy not found. VibeVoice cannot be used.")

# --- CONFIGURATION ---
# Set to True to attempt loading the 0.5B model (Requires torch/GPU/Lightning AI)
TRY_VIBEVOICE = True

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
        # 🟢 FINAL CONFIRMED FIX: Import the exact class name and alias it.
        from VibeVoice.vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference as VibeVoice
        
        logger.info("⏳ Loading VibeVoice Model (0.5B)...")
        
        # Use Accelerator for robust device handling (CPU/GPU detection)
        accelerator = Accelerator()
        
        # Load the model
        model = VibeVoice.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
        
        # Prepare model for the device managed by accelerator
        model = accelerator.prepare(model)
        model.eval() # Set model to evaluation mode
        
        # Verify device
        device = model.device if hasattr(model, 'device') else "CPU (Accelerated)"
        TTS_STRATEGY = f"VibeVoice (Local {device})"
        logger.info(f"✅ VibeVoice Loaded on {device}")
        
    except ImportError as e:
        logger.warning(f"⚠️ VibeVoice library not found: {e}. Ensure the repo is cloned and installed with 'pip install -e VibeVoice'.")
        logger.warning("➡️ System will fallback to EdgeTTS.")
        model = None
    except Exception as e:
        logger.warning(f"⚠️ VibeVoice failed to load, even with dependencies: {e}")
        logger.warning("➡️ System will fallback to EdgeTTS (Cloud).")
        model = None
else:
    if TRY_VIBEVOICE and not TORCH_AVAILABLE:
        logger.warning("⚠️ Cannot try VibeVoice: PyTorch/Accelerate dependencies are missing.")
        logger.warning("➡️ System will fallback to EdgeTTS.")
    model = None


# --- GENERATORS ---

async def generate_vibevoice_stream(text):
    """Generates audio using the local 0.5B model, handling streaming/non-streaming."""
    
    with torch.no_grad():
        if hasattr(model, 'generate_stream'):
            # Use the efficient streaming method if available
            stream = model.generate_stream(text)
            for chunk in stream:
                # Assuming chunk is raw audio bytes/tensor already processed for streaming
                yield chunk
        else:
            # Fallback for non-streaming implementation: Generate full audio and convert
            output = model.generate(text)
            
            # Ensure output is a 1D numpy array
            audio_data = output.cpu().numpy().squeeze()
            if audio_data.ndim > 1:
                 # Flatten if necessary, assuming single channel for TTS
                 audio_data = audio_data.flatten()
            
            # Write to a BytesIO object for in-memory WAV conversion
            byte_io = io.BytesIO()
            # VibeVoice sample rate is typically 24000 Hz
            wav.write(byte_io, 24000, audio_data)
            
            yield byte_io.getvalue()

async def generate_edgetts_stream(text):
    """Fallback: Uses Microsoft Edge Cloud TTS (No GPU required)"""
    # en-US-AriaNeural is a high quality voice, using mp3 format for smaller size
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    
    # edge-tts streams raw audio data chunks
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

# --- ROUTES ---

@app.get("/")
def home():
    """Health check and status endpoint."""
    return {"status": "Online", "strategy": TTS_STRATEGY}

@app.post("/speak")
async def speak(payload: dict = Body(...)):
    """Generates and streams audio for the given text."""
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    logger.info(f"Speaking via {TTS_STRATEGY}: {text[:40]}...")

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
    # Check for hardware environment before running
    if model and TORCH_AVAILABLE and not torch.cuda.is_available():
        logger.warning("Running VibeVoice on CPU will be VERY slow. Performance requires a GPU.")

    uvicorn.run(app, host="0.0.0.0", port=8080)

import os
import io
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceServer")

app = FastAPI()

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

# --- GENERATORS ---

async def generate_vibevoice_stream(text):
    """Generates audio using the local 0.5B model."""
    global model
    if model is None:
        return

    # 1. Attempt Streaming Generation
    try:
        if hasattr(model, 'generate_stream') and callable(model.generate_stream):
            with torch.no_grad():
                stream = model.generate_stream(text)
                for chunk in stream:
                    yield chunk
            return # Exit if streaming worked
    except Exception as e:
        logger.warning(f"Streaming generation failed, falling back to batch: {e}")

    # 2. Batch Generation (Fallback)
    with torch.no_grad():
        output = model.generate(text)

        # Handle complex VibeVoice output objects
        audio_tensor = None
        
        # Check specific attributes known for VibeVoice/HF outputs
        if hasattr(output, 'audio'):
            audio_tensor = output.audio
        elif hasattr(output, 'waveform'):
            audio_tensor = output.waveform
        elif hasattr(output, 'sequences'):
            audio_tensor = output.sequences
        elif isinstance(output, dict) and 'audio' in output:
            audio_tensor = output['audio']
        elif isinstance(output, torch.Tensor):
            audio_tensor = output
        
        # Validation
        if audio_tensor is None:
            logger.error(f"Could not extract audio from model output. Type: {type(output)}")
            raise HTTPException(status_code=500, detail="Model output format not recognized")

        # Conversion: Tensor -> Numpy -> Wav Bytes
        audio_data = audio_tensor.cpu().numpy().squeeze()
        
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        byte_io = io.BytesIO()
        wav.write(byte_io, 24000, audio_data) # 24kHz is standard for VibeVoice
        yield byte_io.getvalue()

async def generate_edgetts_stream(text):
    """Fallback: Uses Microsoft Edge Cloud TTS"""
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    async for chunk in communicate.stream():
        # Linter fix: explicitly check type and key presence
        if chunk["type"] == "audio" and "data" in chunk:
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
        return StreamingResponse(
            generate_vibevoice_stream(text),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"}
        )
    else:
        return StreamingResponse(
            generate_edgetts_stream(text),
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache"}
        )

if __name__ == "__main__":
    if model and TORCH_AVAILABLE and not torch.cuda.is_available():
        logger.warning("Running on CPU. Expect high latency.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
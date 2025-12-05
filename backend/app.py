import os
import io
import logging
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURATION ---
# Set to True to attempt loading the 0.5B model (Requires GPU)
# Set to False to force EdgeTTS (CPU Friendly / Free Tier)
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

if TRY_VIBEVOICE:
    try:
        # We try to import VibeVoice. 
        # On Lightning AI, you must clone the repo into this folder first.
        # Structure assumption: backend/VibeVoice/vibevoice
        from VibeVoice.vibevoice import VibeVoice
        
        logger.info("⏳ Loading VibeVoice Model (0.5B)...")
        model = VibeVoice.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
        # Auto-detect device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info(f"✅ VibeVoice Loaded on {device}")
    except Exception as e:
        logger.warning(f"⚠️ VibeVoice not found or failed to load: {e}")
        logger.warning("➡️ System will fallback to EdgeTTS (Cloud)")

# --- GENERATORS ---

async def generate_vibevoice_stream(text):
    """Generates audio using the local 0.5B model"""
    # Note: If the specific VibeVoice version supports .stream(), use it.
    # Otherwise, we generate full audio and yield bytes.
    # This is a wrapper for compatibility.
    if hasattr(model, 'generate_stream'):
        stream = model.generate_stream(text)
        for chunk in stream:
            yield chunk
    else:
        # Fallback for non-streaming implementation
        with torch.no_grad():
            output = model.generate(text)
            # Convert tensor to wav bytes (simplified)
            import scipy.io.wavfile as wav
            byte_io = io.BytesIO()
            wav.write(byte_io, 24000, output.cpu().numpy())
            yield byte_io.getvalue()

async def generate_edgetts_stream(text):
    """Fallback: Uses Microsoft Edge Cloud TTS (No GPU required)"""
    import edge_tts
    # en-US-AriaNeural is a high quality voice
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

@app.get("/")
def home():
    strategy = "VibeVoice (Local GPU)" if model else "EdgeTTS (Cloud Fallback)"
    return {"status": "Online", "strategy": strategy}

@app.post("/speak")
async def speak(payload: dict = Body(...)):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    logger.info(f"Speaking: {text[:30]}...")

    if model:
        return StreamingResponse(generate_vibevoice_stream(text), media_type="audio/wav")
    else:
        return StreamingResponse(generate_edgetts_stream(text), media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

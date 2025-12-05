# Gemini + VibeVoice Conversational AI

A real-time voice bot using Google Gemini (Brain) and Microsoft VibeVoice (Voice).

## 📂 Project Structure
- **frontend/**: Next.js Chat UI (Deploy to Vercel)
- **backend/**: Python FastAPI Server (Deploy to Lightning AI)

---

## 🚀 Deployment Guide

### Step 1: Backend (The Voice)
1. Go to [Lightning.ai](https://lightning.ai) and create a **CPU Studio** (Free) or **GPU Studio** (Fast).
2. Open the studio terminal and clone your repo:
   `git clone https://github.com/YOUR_GITHUB_USER/gemini-vibevoice-bot.git`
3. Go to the backend folder:
   `cd gemini-vibevoice-bot/backend`
4. Install dependencies:
   `pip install -r requirements.txt`
   `sudo apt-get update && sudo apt-get install -y ffmpeg`
5. (Optional) If you have a GPU and want VibeVoice:
   `git clone https://github.com/microsoft/VibeVoice`
   `pip install -e VibeVoice`
   *(Edit app.py to ensure imports match your structure)*
6. Run the server:
   `python app.py`
7. **Expose Port 8080:** Click the "Ports" plugin in Lightning AI -> Make Port 8080 **Public**.
8. Copy the URL (e.g., `https://your-studio-8080.lightning.ai`).

### Step 2: Frontend (The UI)
1. Push this project to GitHub.
2. Go to [Vercel](https://vercel.com) -> Add New Project -> Import from GitHub.
3. **CRITICAL SETTING:** - In Vercel Project Settings, find **"Root Directory"**.
   - Click Edit and select **`frontend`**.
4. Set Environment Variables in Vercel:
   - `NEXT_PUBLIC_GEMINI_API_KEY`: (Get from Google AI Studio)
   - `NEXT_PUBLIC_TTS_URL`: (Paste your Lightning AI URL)
5. Click **Deploy**.

---

## 🛠 Local Development
1. **Backend:**
   `cd backend && python app.py`
2. **Frontend:**
   - Rename `frontend/.env.local.example` to `.env.local` and add keys.
   - `cd frontend && npm run dev`

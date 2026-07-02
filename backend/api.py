# backend/api.py
"""
FastAPI application for Speech Personality Analysis.

Provides REST endpoints for audio analysis using the
multi-agent AI pipeline with NVIDIA NIM (meta/llama-3.1-70b-instruct).
"""

import os
import shutil
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from link import run_pipeline

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Speech Personality Analysis API",
    description="AI-powered speech analysis with multi-agent personality insights",
    version="2.0.0",
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment readiness."""
    nvidia_configured = bool(os.getenv("NVIDIA_API_KEY"))
    return {
        "status": "healthy",
        "llm_provider": "nvidia-nim",
        "nvidia_configured": nvidia_configured,
        "model": os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-70b-instruct"),
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file for speech and personality insights.
    
    Accepts audio files (WAV, WebM, MP3, etc.) and returns:
    - Transcript
    - Speech metrics
    - Agent analysis results
    - Final personality report
    """
    try:
        # Save uploaded file
        file_ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run the analysis pipeline
        result = run_pipeline(audio_path)
        
        # Clean up uploaded file
        try:
            os.remove(audio_path)
        except OSError:
            pass
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

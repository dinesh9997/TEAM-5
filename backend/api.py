# backend/api.py
"""
FastAPI application for Speech Personality Analysis.

Provides REST endpoints for audio analysis using the
multi-agent AI pipeline with NVIDIA NIM (meta/llama-3.1-70b-instruct).
"""

import os
import shutil
import uuid
import traceback
import logging
import wave
import struct

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def convert_to_wav(input_path: str, output_path: str, target_sr: int = 16000) -> str:
    """
    Convert any audio file (WebM, OGG, MP3, etc.) to a proper 16-bit PCM WAV
    using PyAV. This ensures all downstream components (Silero VAD, openSMILE,
    faster-whisper) receive a format they can natively read.
    """
    import av

    container = av.open(input_path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(
        format="s16",      # 16-bit signed int PCM
        layout="mono",
        rate=target_sr,
    )

    pcm_frames = []
    for frame in container.decode(stream):
        resampled = resampler.resample(frame)
        if resampled:
            for rf in resampled:
                pcm_frames.append(rf.to_ndarray().flatten())

    container.close()

    if not pcm_frames:
        raise ValueError("No audio frames could be decoded from the uploaded file")

    pcm_data = np.concatenate(pcm_frames).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(target_sr)
        wf.writeframes(pcm_data.tobytes())

    logger.info(
        f"Converted {input_path} -> {output_path} "
        f"({len(pcm_data)/target_sr:.1f}s, {target_sr}Hz, 16-bit mono WAV)"
    )
    return output_path


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
    """
    raw_path = None
    wav_path = None
    try:
        # Save uploaded file with its original extension
        file_ext = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
        raw_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")

        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Audio saved: {raw_path} ({os.path.getsize(raw_path)} bytes)")

        # Always convert to a proper WAV so every pipeline component can read it
        wav_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
        convert_to_wav(raw_path, wav_path)

        # Run the analysis pipeline on the converted WAV
        result = run_pipeline(wav_path)
        return result

    except Exception as e:
        # Log the full traceback so we can debug
        logger.error("Pipeline failed with exception:")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Always clean up temporary files
        for path in (raw_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

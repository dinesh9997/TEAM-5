# llm1/llm_config.py
"""
LLM Configuration for Speech Analysis Pipeline.

Uses Google Gemini API. Configure via environment variables or .env file.
Get your FREE API key at: https://aistudio.google.com/apikey
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model settings
LLM_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# llm1/llm_config.py
"""
LLM Configuration for Speech Analysis Pipeline.

Uses NVIDIA NIM API (meta/llama-3.1-70b-instruct).
Configure via environment variables or .env file.
Get your FREE API key at: https://build.nvidia.com/
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NVIDIA NIM API Key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Model settings — llama-3.1-70b gives the best quality output
LLM_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-70b-instruct")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# NVIDIA NIM base URL
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

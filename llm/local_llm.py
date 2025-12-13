"""LLM wrapper used by agents.

This module tries to use the `Ollama` client if available, otherwise
falls back to a lightweight stub that returns deterministic JSON for
testing purposes.
"""
import json

try:
    from langchain_community.llms.ollama import Ollama

    llm = Ollama(
        model="llama3",
        temperature=0.3
    )
except Exception:
    class _StubLLM:
        def invoke(self, prompt: str) -> str:
            p = prompt.lower() if prompt else ""
            if "communication analysis ai agent" in p:
                resp = {
                    "clarity_score": 85,
                    "fluency_level": "Good",
                    "speech_structure": "Structured",
                    "vocabulary_level": "Advanced"
                }
            elif "confidence & emotion analysis ai agent" in p or "confidence & emotion" in p:
                resp = {"confidence_level": "High", "nervousness": "Low", "emotion": "Calm"}
            elif "personality mapping ai agent" in p or "personality" in p:
                resp = {"personality_type": "Balanced", "assertiveness": "Moderate", "expressiveness": "Moderate"}
            else:
                resp = {"message": "stub response", "note": "unrecognized prompt"}
            return json.dumps(resp)

    llm = _StubLLM()
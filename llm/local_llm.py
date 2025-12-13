"""LLM wrapper used by agents.

This module tries to use the `Ollama` client if available, otherwise
falls back to a lightweight stub that returns deterministic JSON for
testing purposes.
"""
import json

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS


class _StubLLM:
    """Fallback LLM that returns deterministic JSON for testing when Ollama is unavailable."""
    
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
        elif "communication coach" in p or "personality report" in p:
            # Final report stub
            return """
📊 **Communication Overview**
- Clarity Score: 85/100 (Good)
- Fluency: Good with structured delivery
- Vocabulary: Advanced level

💪 **Confidence & Emotional Tone**
- Confidence Level: High
- Nervousness: Low
- Emotional State: Calm and composed

🧠 **Personality Insights**
- Type: Balanced communicator
- Assertiveness: Moderate
- Expressiveness: Moderate

⭐ **Key Strengths**
• Clear and structured communication
• Confident delivery with controlled emotions
• Professional and balanced approach

🎯 **Improvement Recommendations**
• Continue practicing for even more natural flow
• Consider adding more vocal variety for engagement
• Maintain current confident pace

*Note: This is a stub response - Ollama server is not running.*
"""
        else:
            resp = {"message": "stub response", "note": "Ollama not running - using fallback"}
        return json.dumps(resp)


class _LazyOllamaLLM:
    """Lazy-loading wrapper that tries Ollama first, falls back to stub."""
    
    def __init__(self):
        self._llm = None
        self._initialized = False
    
    def _get_llm(self):
        if not self._initialized:
            self._initialized = True
            try:
                # Try new package first
                try:
                    from langchain_ollama import OllamaLLM
                    self._llm = OllamaLLM(
                        model=LLM_MODEL_NAME, 
                        temperature=TEMPERATURE,
                        num_predict=MAX_TOKENS
                    )
                except ImportError:
                    from langchain_community.llms.ollama import Ollama
                    self._llm = Ollama(
                        model=LLM_MODEL_NAME, 
                        temperature=TEMPERATURE,
                        num_predict=MAX_TOKENS
                    )
                
                # Test connection
                self._llm.invoke("test")
            except Exception as e:
                print(f"⚠️ Ollama not available ({e}), using stub LLM")
                self._llm = _StubLLM()
        return self._llm
    
    def invoke(self, prompt: str) -> str:
        return self._get_llm().invoke(prompt)


# Export lazy-loading LLM instance
llm = _LazyOllamaLLM()
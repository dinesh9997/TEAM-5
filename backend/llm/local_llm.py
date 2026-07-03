"""LLM wrapper used by agents.

Uses NVIDIA NIM API (meta/llama-3.1-70b-instruct) via langchain-nvidia-ai-endpoints.
Falls back to a deterministic stub for testing when the API is unavailable.
"""
import json

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS, NVIDIA_API_KEY, NVIDIA_BASE_URL


class _StubLLM:
    """Fallback LLM that returns deterministic JSON for testing when NVIDIA API is unavailable."""

    def invoke(self, prompt: str) -> str:
        p = prompt.lower() if prompt else ""
        if "communication analysis ai agent" in p or "senior communication skills analyst" in p:
            resp = {
                "communication_score": 85,
                "clarity_level": "High",
                "fluency_level": "High",
                "speech_pacing": "Balanced",
                "key_observations": ["Clear articulation", "Well-structured sentences"],
                "communication_strengths": ["Good vocabulary", "Logical flow"],
                "communication_gaps": ["Minor filler words"],
                "improvement_suggestions": ["Reduce filler words", "Add more pauses for emphasis"]
            }
        elif "confidence" in p and "emotion" in p:
            resp = {
                "confidence_score": 78,
                "confidence_level": "High",
                "emotional_tone": "Positive",
                "vocal_energy_assessment": "Moderate",
                "confidence_indicators": ["Steady pace", "Clear projection"],
                "possible_challenges": ["Slight pitch variation under stress"],
                "confidence_enhancement_tips": ["Practice deep breathing", "Maintain consistent volume"]
            }
        elif "personality" in p:
            resp = {
                "personality_type": "Ambivert",
                "interaction_style": "Balanced",
                "professional_presence": "Competent",
                "key_personality_traits": ["Analytical", "Composed"],
                "strengths_in_interaction": ["Active listening", "Structured responses"],
                "growth_opportunities": ["More vocal expressiveness", "Strategic storytelling"],
                "overall_summary": "A balanced communicator with strong analytical tendencies."
            }
        elif "communication coach" in p or "report" in p:
            return (
                "📊 **Communication Overview**\n"
                "- Clarity: High | Fluency: High | Pacing: Balanced\n\n"
                "💪 **Confidence & Emotional Tone**\n"
                "- Confidence: High | Emotional Tone: Positive\n\n"
                "🧠 **Personality Insights**\n"
                "- Type: Ambivert | Presence: Competent\n\n"
                "⭐ **Key Strengths**\n"
                "• Clear structured communication\n"
                "• Confident, positive delivery\n\n"
                "🎯 **Improvement Recommendations**\n"
                "• Configure NVIDIA_API_KEY in .env for real AI-powered analysis\n\n"
                "*Note: Stub response — set NVIDIA_API_KEY in .env for full analysis.*"
            )
        else:
            resp = {"message": "stub response", "note": "NVIDIA API not configured — using fallback"}
        return json.dumps(resp)


class _LazyNvidiaLLM:
    """Lazy-loading wrapper that uses NVIDIA NIM API, falls back to stub."""

    def __init__(self):
        self._llm = None
        self._initialized = False

    def _get_llm(self):
        if not self._initialized:
            self._initialized = True

            if not NVIDIA_API_KEY:
                print("[WARNING] NVIDIA_API_KEY not set. Using stub LLM.")
                print("   Get your free key at: https://build.nvidia.com/")
                print("   Then set it in backend/.env file")
                self._llm = _StubLLM()
                return self._llm

            try:
                from langchain_nvidia_ai_endpoints import ChatNVIDIA

                self._llm = ChatNVIDIA(
                    model=LLM_MODEL_NAME,
                    nvidia_api_key=NVIDIA_API_KEY,
                    base_url=NVIDIA_BASE_URL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                # Test connection
                test_response = self._llm.invoke("Say ok in one word.")
                if hasattr(test_response, "content"):
                    _ = test_response.content
                print(f"[OK] NVIDIA NIM ({LLM_MODEL_NAME}) connected successfully")

            except Exception as e:
                print(f"[WARNING] NVIDIA API not available ({e}), using stub LLM")
                self._llm = _StubLLM()

        return self._llm

    def invoke(self, prompt: str) -> str:
        llm = self._get_llm()
        response = llm.invoke(prompt)
        # ChatNVIDIA returns AIMessage — extract text content
        if hasattr(response, "content"):
            return response.content
        return str(response)


# Export lazy-loading LLM instance
llm = _LazyNvidiaLLM()
# llm1/local_llm.py
"""
LLM provider for report generation.

Returns a NVIDIA NIM LLM instance via langchain-nvidia-ai-endpoints.
Falls back to stub if the API is unavailable.
"""

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS, NVIDIA_API_KEY, NVIDIA_BASE_URL


class _StubLLM:
    """Fallback stub LLM for when NVIDIA API is unavailable."""

    def invoke(self, prompt: str) -> str:
        return (
            "📊 **Communication Overview**\n"
            "- Analysis based on stub data (NVIDIA API not configured)\n\n"
            "💪 **Confidence & Emotional Tone**\n"
            "- Confidence: Moderate\n\n"
            "🧠 **Personality Insights**\n"
            "- Type: Balanced communicator\n\n"
            "⭐ **Key Strengths**\n"
            "• Structured communication\n\n"
            "🎯 **Improvement Recommendations**\n"
            "• Set NVIDIA_API_KEY in backend/.env for real AI-powered analysis\n\n"
            "*Note: Stub response — NVIDIA_API_KEY not configured.*"
        )


def get_llm():
    """
    Returns a NVIDIA NIM LLM instance (meta/llama-3.1-70b-instruct).
    Falls back to stub if the API key is not configured or connection fails.
    """
    if not NVIDIA_API_KEY:
        print("⚠️  NVIDIA_API_KEY not set. Using stub LLM for report generation.")
        print("   Get your free key at: https://build.nvidia.com/")
        return _StubLLM()

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        llm = ChatNVIDIA(
            model=LLM_MODEL_NAME,
            nvidia_api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # Quick test to verify the API key works
        test_response = llm.invoke("Say ok in one word.")
        if hasattr(test_response, "content"):
            _ = test_response.content

        print(f"✅ NVIDIA NIM ({LLM_MODEL_NAME}) ready for report generation")
        return llm

    except Exception as e:
        print(f"⚠️  NVIDIA API not available: {e}")
        print("   Using stub LLM for testing...")
        return _StubLLM()

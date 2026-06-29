# llm1/local_llm.py
"""
LLM provider for report generation.

Returns a Google Gemini LLM instance via langchain-google-genai.
Falls back to stub if the API is unavailable.
"""

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS, GEMINI_API_KEY


class _StubLLM:
    """Fallback stub LLM for when Gemini API is unavailable."""
    
    def invoke(self, prompt: str) -> str:
        return (
            "📊 **Communication Overview**\n"
            "- Analysis based on stub data (Gemini API not configured)\n\n"
            "💪 **Confidence & Emotional Tone**\n"
            "- Confidence: Moderate\n\n"
            "🧠 **Personality Insights**\n"
            "- Type: Balanced communicator\n\n"
            "⭐ **Key Strengths**\n"
            "• Structured communication\n\n"
            "🎯 **Improvement Recommendations**\n"
            "• Configure Gemini API for real AI-powered analysis\n\n"
            "*Note: Stub response — set GEMINI_API_KEY in .env for real analysis.*"
        )


def get_llm():
    """
    Returns a Google Gemini LLM instance for report generation.
    Falls back to stub if the API key is not configured.
    """
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not set. Using stub LLM for report generation.")
        print("   Get your free API key at: https://aistudio.google.com/apikey")
        return _StubLLM()
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
        
        # Quick test to verify the API key works
        test_response = llm.invoke("Say 'ok' in one word.")
        if hasattr(test_response, 'content'):
            _ = test_response.content
        
        print(f"✅ Gemini LLM ({LLM_MODEL_NAME}) ready for report generation")
        return llm
        
    except Exception as e:
        print(f"⚠️ Gemini API not available: {e}")
        print("   Using stub LLM for testing...")
        return _StubLLM()

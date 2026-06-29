"""LLM wrapper used by agents.

This module provides the LLM instance used by all agents in the pipeline.
Uses Google Gemini API via langchain-google-genai for cloud-based inference.
Falls back to a deterministic stub for testing when the API is unavailable.
"""
import json

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS, GEMINI_API_KEY


class _StubLLM:
    """Fallback LLM that returns deterministic JSON for testing when Gemini API is unavailable."""
    
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
        elif "confidence & emotion analysis ai agent" in p or "confidence & emotion" in p or "voice confidence analyst" in p:
            resp = {
                "confidence_score": 78,
                "confidence_level": "High",
                "emotional_tone": "Positive",
                "vocal_energy_assessment": "Moderate",
                "confidence_indicators": ["Steady pace", "Clear projection"],
                "possible_challenges": ["Slight pitch variation under stress"],
                "confidence_enhancement_tips": ["Practice deep breathing", "Maintain consistent volume"]
            }
        elif "personality mapping ai agent" in p or "personality" in p:
            resp = {
                "personality_type": "Ambivert",
                "interaction_style": "Balanced",
                "professional_presence": "Competent",
                "key_personality_traits": ["Analytical", "Composed"],
                "strengths_in_interaction": ["Active listening", "Structured responses"],
                "growth_opportunities": ["More vocal expressiveness", "Strategic storytelling"],
                "overall_summary": "A balanced communicator with strong analytical tendencies."
            }
        elif "communication coach" in p or "personality report" in p:
            # Final report stub
            return """
📊 **Communication Overview**
- Clarity Score: 85/100 (High)
- Fluency: High with balanced pacing
- Vocabulary: Advanced level

💪 **Confidence & Emotional Tone**
- Confidence Level: High
- Emotional Tone: Positive
- Vocal Energy: Moderate

🧠 **Personality Insights**
- Type: Ambivert communicator
- Interaction Style: Balanced
- Professional Presence: Competent

⭐ **Key Strengths**
• Clear and structured communication
• Confident delivery with positive tone
• Professional and balanced approach

🎯 **Improvement Recommendations**
• Reduce filler words for smoother flow
• Add strategic pauses for emphasis
• Experiment with vocal variety for engagement

*Note: This is a stub response — Gemini API key not configured.*
"""
        else:
            resp = {"message": "stub response", "note": "Gemini API not configured — using fallback"}
        return json.dumps(resp)


class _LazyGeminiLLM:
    """Lazy-loading wrapper that tries Google Gemini API first, falls back to stub."""
    
    def __init__(self):
        self._llm = None
        self._initialized = False
    
    def _get_llm(self):
        if not self._initialized:
            self._initialized = True
            
            if not GEMINI_API_KEY:
                print("⚠️ GEMINI_API_KEY not set. Using stub LLM.")
                print("   Get your free API key at: https://aistudio.google.com/apikey")
                print("   Then set it in backend/.env file")
                self._llm = _StubLLM()
                return self._llm
            
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                self._llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL_NAME,
                    google_api_key=GEMINI_API_KEY,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                )
                
                # Test connection
                test_response = self._llm.invoke("Say 'ok' in one word.")
                # ChatGoogleGenerativeAI returns AIMessage, extract text
                if hasattr(test_response, 'content'):
                    _ = test_response.content
                print(f"✅ Google Gemini ({LLM_MODEL_NAME}) connected successfully")
                
            except Exception as e:
                print(f"⚠️ Gemini API not available ({e}), using stub LLM")
                self._llm = _StubLLM()
        
        return self._llm
    
    def invoke(self, prompt: str) -> str:
        llm = self._get_llm()
        response = llm.invoke(prompt)
        # ChatGoogleGenerativeAI returns AIMessage object; extract text content
        if hasattr(response, 'content'):
            return response.content
        return str(response)


# Export lazy-loading LLM instance
llm = _LazyGeminiLLM()
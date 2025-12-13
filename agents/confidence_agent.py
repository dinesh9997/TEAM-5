from llm_helper import llm
from llm1.prompt_templates import CONFIDENCE_PROMPT
from utils.parser import safe_parse

# Import RAG for context augmentation
try:
    from rag.retriever import get_retriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Import GuardrailsAI for output validation
try:
    from guardrails_config import validate_agent_response
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    def validate_agent_response(x, _): return x


def _get_confidence_context(state):
    """Retrieve relevant confidence knowledge from RAG"""
    if not RAG_AVAILABLE:
        return ""
    try:
        retriever = get_retriever()
        f = state.get("audio_features", {})
        # Use all relevant vocal metrics for better semantic retrieval
        metrics = {
            "energy_level": f.get("energy_level", ""),
            "pitch_variance": f.get("pitch_variance", ""),
            "pause_ratio": f.get("pause_ratio", "")
        }
        return retriever.get_context_for_analysis("confidence", metrics)
    except Exception as e:
        print(f"⚠️ Confidence RAG context failed: {e}")
        return ""


def confidence_agent(state):
    try:
        f = state["audio_features"]
        
        # Get RAG context
        rag_context = _get_confidence_context(state)
        
        # Build prompt using template
        prompt = CONFIDENCE_PROMPT.format(
            rag_context=f"EXPERT KNOWLEDGE:\n{rag_context}\n" if rag_context else "",
            pitch_variance=f.get("pitch_variance", "N/A"),
            energy_level=f.get("energy_level", "N/A"),
            pause_ratio=f.get("pause_ratio", "N/A")
        )

        response = llm.invoke(prompt)
        parsed = safe_parse(response)
        
        # Validate output with guardrails
        validated = validate_agent_response(parsed, "confidence_agent")

        return {"confidence_emotion_analysis": validated}

    except Exception as e:
        return {
            "confidence_emotion_analysis": {
                "error": str(e),
                "status": "failed"
            }
        }

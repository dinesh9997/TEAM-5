from llm_helper import llm
from llm1.prompt_templates import COMMUNICATION_PROMPT
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


def _get_communication_context(state):
    """Retrieve relevant communication knowledge from RAG"""
    if not RAG_AVAILABLE:
        return ""
    try:
        retriever = get_retriever()
        f = state.get("audio_features", {})
        # Use all relevant metrics for better semantic retrieval
        metrics = {
            "speech_rate": f.get("speech_rate", ""),
            "pause_ratio": f.get("pause_ratio", ""),
            "transcript_preview": state.get("transcript", "")[:100]
        }
        return retriever.get_context_for_analysis("communication", metrics)
    except Exception as e:
        print(f"⚠️ Communication RAG context failed: {e}")
        return ""


def communication_agent(state):
    try:
        transcript = state["transcript"]
        f = state["audio_features"]
        
        # Get RAG context
        rag_context = _get_communication_context(state)
        
        # Build prompt using template
        prompt = COMMUNICATION_PROMPT.format(
            rag_context=f"EXPERT KNOWLEDGE:\n{rag_context}\n" if rag_context else "",
            transcript=transcript[:500] if len(transcript) > 500 else transcript,  # Limit for efficiency
            speech_rate=f.get("speech_rate", "N/A"),
            pause_ratio=f.get("pause_ratio", "N/A")
        )

        response = llm.invoke(prompt)
        parsed = safe_parse(response)
        
        # Validate output with guardrails
        validated = validate_agent_response(parsed, "communication_agent")

        return {"communication_analysis": validated}

    except Exception as e:
        return {
            "communication_analysis": {
                "error": str(e),
                "status": "failed"
            }
        }

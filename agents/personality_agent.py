from llm_helper import llm
from llm1.prompt_templates import PERSONALITY_PROMPT
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


def _get_personality_context(state):
    """Retrieve relevant personality knowledge from RAG"""
    if not RAG_AVAILABLE:
        return ""
    try:
        retriever = get_retriever()
        # Extract key traits from previous analyses for better retrieval
        comm = state.get("communication_analysis", {})
        conf = state.get("confidence_emotion_analysis", {})
        metrics = {
            "fluency_level": comm.get("fluency_level", "") if isinstance(comm, dict) else "",
            "confidence_level": conf.get("confidence_level", "") if isinstance(conf, dict) else "",
            "emotion": conf.get("emotion", "") if isinstance(conf, dict) else ""
        }
        return retriever.get_context_for_analysis("personality", metrics)
    except Exception as e:
        print(f"⚠️ Personality RAG context failed: {e}")
        return ""


def personality_agent(state):
    try:
        # Get RAG context with state for better retrieval
        rag_context = _get_personality_context(state)
        
        # Build prompt using template
        prompt = PERSONALITY_PROMPT.format(
            rag_context=f"EXPERT KNOWLEDGE:\n{rag_context}\n" if rag_context else "",
            communication_analysis=state.get("communication_analysis", "N/A"),
            confidence_analysis=state.get("confidence_emotion_analysis", "N/A")
        )

        response = llm.invoke(prompt)
        parsed = safe_parse(response)
        
        # Validate output with guardrails
        validated = validate_agent_response(parsed, "personality_agent")

        return {"personality_analysis": validated}

    except Exception as e:
        return {
            "personality_analysis": {
                "error": str(e),
                "status": "failed"
            }
        }

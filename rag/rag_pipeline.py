from rag.retriever import get_retriever
from llm1.local_llm import get_llm
from llm1.prompt_templates import REPORT_PROMPT

# Import GuardrailsAI for report validation
try:
    from guardrails_config import validate_final_report
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    def validate_final_report(x): return x


def rag_enhanced_report(agent_outputs: dict) -> str:
    """
    Generate a RAG-enhanced report using retrieved knowledge.
    Uses the custom RAGRetriever API (not LangChain's invoke).
    """
    retriever = get_retriever()
    llm = get_llm()
    
    # Extract analysis results to identify weak areas for targeted improvements
    comm = agent_outputs.get("communication_analysis", {})
    conf = agent_outputs.get("confidence_emotion_analysis", {})
    pers = agent_outputs.get("personality_analysis", {})
    
    # Identify areas needing improvement based on agent outputs
    weak_areas = []
    
    if isinstance(comm, dict):
        if comm.get("clarity_score", 100) < 70:
            weak_areas.append("clarity")
        fluency = str(comm.get("fluency_level", "")).lower()
        if fluency in ["poor", "average"]:
            weak_areas.append("fluency")
        structure = str(comm.get("speech_structure", "")).lower()
        if structure in ["disorganized", "basic"]:
            weak_areas.append("speech structure")
            
    if isinstance(conf, dict):
        confidence = str(conf.get("confidence_level", "")).lower()
        if confidence == "low":
            weak_areas.append("confidence")
        nervousness = str(conf.get("nervousness", "")).lower()
        if nervousness in ["high", "medium"]:
            weak_areas.append("nervousness reduction")
            
    if isinstance(pers, dict):
        assertiveness = str(pers.get("assertiveness", "")).lower()
        if assertiveness == "low":
            weak_areas.append("assertiveness")
    
    # Get targeted improvement recommendations using RAGRetriever's method
    improve_metrics = {"weak_areas": weak_areas if weak_areas else ["general speaking skills"]}
    rag_context = retriever.get_context_for_analysis("improvement", improve_metrics)
    
    # Build prompt using template
    prompt = REPORT_PROMPT.format(
        rag_context=rag_context if rag_context else "No specific recommendations available.",
        agent_outputs=agent_outputs
    )

    report = llm.invoke(prompt)
    
    # Validate final report with guardrails
    validated_report = validate_final_report(report)
    
    return validated_report

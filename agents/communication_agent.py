from llm_helper import llm
from utils.parser import safe_parse

def communication_agent(state):
    try:
        transcript = state["transcript"]
        f = state["audio_features"]

        prompt = f"""
You are a Communication Analysis AI Agent.

ROLE:
Analyze spoken communication objectively using measurable indicators.

INPUT:
Transcript:
{transcript}

Speech Metrics:
- Speech rate: {f.get("speech_rate")} words/min
- Pause ratio: {f.get("pause_ratio")}

ANALYSIS RULES:
- Speech rate 120–160 → normal
- Pause ratio > 0.25 → reduced fluency
- Clear sentence transitions → higher clarity
- Avoid subjective opinions

TASK:
Evaluate communication quality ONLY.

OUTPUT JSON ONLY:
{{
  "clarity_score": number (0–100),
  "fluency_level": "Poor | Average | Good | Excellent",
  "speech_structure": "Disorganized | Basic | Structured | Well-structured",
  "vocabulary_level": "Basic | Intermediate | Advanced"
}}
"""


        response = llm.invoke(prompt)
        parsed = safe_parse(response)

        return {"communication_analysis": parsed}

    except Exception as e:
        return {
            "communication_analysis": {
                "error": str(e),
                "status": "failed"
            }
        }

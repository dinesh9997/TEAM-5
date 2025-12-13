from llm_helper import llm
from utils.parser import safe_parse

def confidence_agent(state):
    try:
        f = state["audio_features"]

        prompt = f"""
You are a Confidence & Emotion Analysis AI Agent.

ROLE:
Infer confidence from vocal delivery, not content meaning.

INPUT FEATURES:
- Pitch variance: {f.get("pitch_variance")}
- Energy level: {f.get("energy_level")}
- Pause ratio: {f.get("pause_ratio")}

REFERENCE RULES:
- Low pitch variance + long pauses → cautious or nervous
- Stable energy → emotional control
- High pauses + medium energy → moderate confidence

TASK:
Infer confidence and emotional tone conservatively.

OUTPUT JSON ONLY:
{{
  "confidence_level": "Low | Moderate | High",
  "nervousness": "Low | Medium | High",
  "emotion": "Calm | Engaged | Cautious | Nervous"
}}
"""


        response = llm.invoke(prompt)
        parsed = safe_parse(response)

        return {"confidence_emotion_analysis": parsed}

    except Exception as e:
        return {
            "confidence_emotion_analysis": {
                "error": str(e),
                "status": "failed"
            }
        }

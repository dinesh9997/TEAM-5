from llm_helper import llm
from utils.parser import safe_parse

def personality_agent(state):
    try:
        prompt = f"""
You are a Personality Mapping AI Agent.

ROLE:
Map communication behavior to personality traits (non-diagnostic).

INPUT:
Communication Analysis:
{state["communication_analysis"]}

Confidence & Emotion Analysis:
{state["confidence_emotion_analysis"]}

MAPPING RULES:
- Moderate confidence + controlled emotion → balanced personality
- Low expressiveness → introverted tendency
- High clarity + moderate assertiveness → professional communicator

TASK:
Map to communication-oriented personality traits only.

OUTPUT JSON ONLY:
{{
  "personality_type": "Introvert | Balanced | Extrovert",
  "assertiveness": "Low | Moderate | High",
  "expressiveness": "Low | Moderate | High"
}}
"""


        response = llm.invoke(prompt)
        parsed = safe_parse(response)

        return {"personality_analysis": parsed}

    except Exception as e:
        return {
            "personality_analysis": {
                "error": str(e),
                "status": "failed"
            }
        }

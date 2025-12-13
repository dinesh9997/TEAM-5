"""Agent orchestrator.

Provides `run_agents(state)` which sequentially calls all agents in `/agents`
and returns a combined analysis dictionary.
"""

import json
from agents.communication_agent import communication_agent
from agents.confidence_agent import confidence_agent
from agents.personality_agent import personality_agent


def run_agents(state):
    """Run communication, confidence, and personality agents in sequence.

    Args:
        state (dict): Pipeline output with `transcript` and `audio_features` keys.

    Returns:
        dict: Combined results with keys `communication_analysis`,
              `confidence_emotion_analysis`, and `personality_analysis`.
    """
    try:
        # Communication analysis (needs transcript + audio features)
        comm_res = communication_agent(state)
        comm = comm_res.get("communication_analysis") if isinstance(comm_res, dict) else None

        # Attach intermediate result for downstream agents
        state_with_comm = dict(state)
        if comm is not None:
            state_with_comm["communication_analysis"] = comm

        # Confidence & emotion analysis
        conf_res = confidence_agent(state_with_comm)
        conf = conf_res.get("confidence_emotion_analysis") if isinstance(conf_res, dict) else None

        # Attach confidence for personality agent
        state_with_comm_conf = dict(state_with_comm)
        if conf is not None:
            state_with_comm_conf["confidence_emotion_analysis"] = conf

        # Personality mapping
        person_res = personality_agent(state_with_comm_conf)
        person = person_res.get("personality_analysis") if isinstance(person_res, dict) else None

        combined = {}
        if comm is not None:
            combined["communication_analysis"] = comm
        else:
            combined["communication_analysis"] = comm_res

        if conf is not None:
            combined["confidence_emotion_analysis"] = conf
        else:
            combined["confidence_emotion_analysis"] = conf_res

        if person is not None:
            combined["personality_analysis"] = person
        else:
            combined["personality_analysis"] = person_res

        return combined

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


if __name__ == "__main__":
    sample_state = {
        "transcript": "I am confident in my ability to communicate effectively.",
        "audio_features": {
            "speech_rate": 130,
            "pitch_variance": 22.5,
            "pause_ratio": 0.18,
            "energy_level": "medium-high"
        }
    }

    result = run_agents(sample_state)

    print("\n🧠 AGENT ORCHESTRATION OUTPUTS\n")
    print("📌 Communication Analysis")
    print(json.dumps(result.get("communication_analysis"), indent=2))

    print("\n📌 Confidence & Emotion Analysis")
    print(json.dumps(result.get("confidence_emotion_analysis"), indent=2))

    print("\n📌 Personality Mapping")
    print(json.dumps(result.get("personality_analysis"), indent=2))

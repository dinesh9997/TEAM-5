from speech_to_text import transcribe_audio
from speech_features import analyze_speech


def get_pipeline_output(audio_file):
    """Run transcription and analysis, return structured data.

    Returns a dict with `transcript` and `audio_features` keys
    matching the requested format.
    """
    # Step 1: Transcription
    transcription_data = transcribe_audio(audio_file)

    # Step 2: Speech Analysis
    results, score, label, wpm, avg_pause = analyze_speech(
        audio_file,
        transcription_data["word_segments"]
    )

    # Assemble the output data structure
    data = {
        "transcript": transcription_data["transcript"],
        "audio_features": {
            "speech_rate": results.get("speech_rate", wpm),
            "pitch_variance": results.get("pitch_variance"),
            "pause_ratio": results.get("pause_ratio", round(avg_pause / (wpm/60) if wpm else 0, 2)),
            "energy_level": results.get("energy_level")
        }
    }

    # Include internal analysis if needed for debugging
    data["_analysis_details"] = {
        "results": results,
        "confidence_score": score,
        "label": label,
        "wpm": wpm,
        "avg_pause": avg_pause
    }

    return data


if __name__ == "__main__":
    AUDIO_FILE = "clean_audio.wav"
    print("🔄 Starting Speech Analysis Pipeline...\n")

    out = get_pipeline_output(AUDIO_FILE)

    print("\n📝 Transcript:\n")
    print(out["transcript"])

    print("\n📊 SPEECH ANALYSIS REPORT")
    for k, v in out["_analysis_details"]["results"].items():
        print(f"{k}: {v}")

    print("\nFormatted Data Return:\n")
    print(out)

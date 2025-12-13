from faster_whisper import WhisperModel

AUDIO_FILE = "clean_audio.wav"

def transcribe_audio(audio_file):
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    print("🎧 Transcribing...")
    segments, info = model.transcribe(audio_file, language="en")

    full_text = ""
    segment_data = []
    word_segments = []

    for seg in segments:
        full_text += seg.text + " "

        segment_data.append({
            "text": seg.text,
            "start": seg.start,
            "end": seg.end
        })

        # ---- Word-level estimation ----
        words = seg.text.strip().split()
        if not words:
            continue

        duration = seg.end - seg.start
        avg_word_time = duration / len(words)

        for i, word in enumerate(words):
            word_start = seg.start + i * avg_word_time
            word_end = word_start + avg_word_time

            word_segments.append({
                "word": word,
                "start": round(word_start, 2),
                "end": round(word_end, 2)
            })

    return {
        "transcript": full_text.strip(),
        "segments": segment_data,
        "word_segments": word_segments
    }


# For standalone testing
if __name__ == "__main__":
    data = transcribe_audio(AUDIO_FILE)
    print("\n📝 Transcript:\n")
    print(data["transcript"]) 

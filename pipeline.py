from speech_to_text import transcribe_audio
from speech_features import analyze_speech

AUDIO_FILE = "clean_audio.wav"

print("🔄 Starting Speech Analysis Pipeline...\n")

# Step 1: Transcription
data = transcribe_audio(AUDIO_FILE)

print("\n📝 Transcript:\n")
print(data["full_text"])

# Step 2: Speech Analysis
print("\n📊 Analyzing speech...\n")
results, score, label, wpm, avg_pause = analyze_speech(
    AUDIO_FILE,
    data["word_segments"]
)

# Step 3: Results
print("===== SPEECH ANALYSIS REPORT =====")
for k, v in results.items():
    print(f"{k}: {v}")

print("\nWords Per Minute:", wpm)
print("Average Pause:", avg_pause)
print("Overall Score:", score)
print("Confidence Level:", label)

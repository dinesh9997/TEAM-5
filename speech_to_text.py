from faster_whisper import WhisperModel

AUDIO_FILE = "clean_audio.wav"

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("🎧 Transcribing...")
segments, info = model.transcribe(AUDIO_FILE, language="en")

full_text = ""

for segment in segments:
    full_text += segment.text + " "

print("\n📝 Transcript:\n")
print(full_text.strip())

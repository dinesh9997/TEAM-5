import librosa
import soundfile as sf
from utils.audio_loader import load_audio

INPUT_AUDIO = "raw_audio.wav"
OUTPUT_AUDIO = "clean_audio.wav"

def preprocess_audio(input_path, output_path):
    # Load audio using PyAV to handle WebM/various formats
    y, sr = load_audio(input_path, target_sr=16000)

    # Normalize volume
    y = librosa.util.normalize(y)

    # Remove silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # Save cleaned audio
    sf.write(output_path, y_trimmed, sr)

    duration = librosa.get_duration(y=y_trimmed, sr=sr)

    print("✅ Audio preprocessing complete")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample Rate: {sr}")

if __name__ == "__main__":
    preprocess_audio(INPUT_AUDIO, OUTPUT_AUDIO)

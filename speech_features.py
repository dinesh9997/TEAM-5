import librosa
import numpy as np

def analyze_speech(audio_file, word_segments):
    y, sr = librosa.load(audio_file)

    # -----------------------
    # Words Per Minute (WPM)
    # -----------------------
    total_words = len(word_segments)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    wpm = round((total_words / duration_sec) * 60, 2)

    # -----------------------
    # Pause Analysis
    # -----------------------
    pauses = []
    for i in range(1, len(word_segments)):
        pause = word_segments[i]["start"] - word_segments[i-1]["end"]
        if pause > 0:
            pauses.append(pause)

    avg_pause = round(np.mean(pauses), 2) if pauses else 0

    # -----------------------
    # Energy (Confidence)
    # -----------------------
    rms = librosa.feature.rms(y=y)[0]
    energy_score = round(np.mean(rms) * 100, 2)

    # -----------------------
    # Pitch (Expressiveness)
    # -----------------------
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = round(np.mean(pitch_values), 2) if len(pitch_values) else 0
    pitch_variance = round(np.var(pitch_values), 2) if len(pitch_values) else 0

    # -----------------------
    # Pause ratio -> total pause time / duration
    # -----------------------
    total_pause_time = round(sum(pauses), 2) if pauses else 0
    pause_ratio = round(total_pause_time / duration_sec, 2) if duration_sec > 0 else 0

    # -----------------------
    # Confidence Score
    # -----------------------
    confidence_score = round(
        (energy_score * 0.4) +
        (min(wpm, 160) * 0.4) +
        ((100 - avg_pause * 20) * 0.2),
        2
    )

    label = (
        "High Confidence" if confidence_score >= 75 else
        "Moderate Confidence" if confidence_score >= 50 else
        "Low Confidence"
    )

    # -----------------------
    # Energy level mapping
    # -----------------------
    if energy_score >= 70:
        energy_level = "high"
    elif energy_score >= 40:
        energy_level = "medium-high"
    elif energy_score >= 20:
        energy_level = "medium"
    else:
        energy_level = "low"

    results = {
        "speech_rate": round(wpm),
        "pitch_variance": pitch_variance,
        "pause_ratio": pause_ratio,
        "energy_level": energy_level,
        "Energy Score": energy_score,
        "Pitch Mean (Hz)": pitch_mean,
        "Speech Duration (sec)": round(duration_sec, 2),
        "Total Words": total_words
    }

    return results, confidence_score, label, wpm, avg_pause

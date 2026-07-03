import librosa
import opensmile
import torch
from utils.audio_loader import load_audio

# ---------------------------
# LOAD MODELS ONCE
# ---------------------------

# openSMILE feature extractor (standardized acoustic features)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ---------------------------
# Silero VAD (Offline, Stable)
# ---------------------------
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True
)

(
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks
) = vad_utils


def compute_pause_ratio(audio_path, sampling_rate=16000):
    """
    Computes pause ratio using Silero VAD.
    pause_ratio = non-speech duration / total duration.
    Uses PyAV-based load_audio instead of Silero's read_audio to avoid
    torchcodec/FFmpeg dependency issues on Windows.
    """
    # Load audio as numpy array, then convert to torch tensor for Silero VAD
    y, sr = load_audio(audio_path, target_sr=sampling_rate)
    wav = torch.from_numpy(y).float()

    # Silero VAD expects values in [-1, 1] range
    if wav.abs().max() > 1.0:
        wav = wav / wav.abs().max()

    speech_timestamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=sampling_rate
    )

    if not speech_timestamps:
        return 1.0, 0.0  # all pause

    speech_time = sum(
        (seg["end"] - seg["start"]) / sampling_rate
        for seg in speech_timestamps
    )

    total_duration = len(wav) / sampling_rate
    pause_time = max(total_duration - speech_time, 0)

    pause_ratio = pause_time / total_duration if total_duration > 0 else 0
    return round(pause_ratio, 2), round(pause_time, 2)



# ---------------------------
# MAIN FUNCTION
# ---------------------------
def analyze_speech(audio_file, word_segments):
    # Load audio using PyAV to handle WebM/various container formats
    y, sr = load_audio(audio_file, target_sr=16000)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    # -----------------------
    # Speech Rate (WPM)
    # -----------------------
    total_words = len(word_segments)
    wpm = round((total_words / duration_sec) * 60, 2) if duration_sec > 0 else 0

    # -----------------------
    # Pause Analysis (Silero VAD)
    # -----------------------
    pause_ratio, total_pause_time = compute_pause_ratio(audio_file)

    # -----------------------
    # Acoustic Features (openSMILE)
    # -----------------------
    features = smile.process_file(audio_file)

    def get_feature(df, name_candidates, default=0.0):
        for name in name_candidates:
            if name in df.columns:
                return float(df[name].iloc[0])
        return default

    loudness = get_feature(
        features,
        ["loudness_sma3_amean", "loudness_sma3_mean"]
    )

    pitch_mean = get_feature(
        features,
        ["F0semitoneFrom27.5Hz_sma3nz_amean"]
    )

    pitch_variance = get_feature(
        features,
        ["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"]
    )

    jitter = get_feature(
        features,
        ["jitterLocal_sma3nz_amean"]
    )

    shimmer = get_feature(
        features,
        ["shimmerLocaldB_sma3nz_amean"]
    )


    # -----------------------
    # Energy Level Mapping
    # -----------------------
    if loudness >= 0.8:
        energy_level = "high"
    elif loudness >= 0.5:
        energy_level = "medium-high"
    elif loudness >= 0.3:
        energy_level = "medium"
    else:
        energy_level = "low"

    # -----------------------
    # Confidence Score
    # -----------------------
    confidence_score = round(
        (min(wpm, 160) / 160) * 40 +
        (loudness * 40) +
        ((1 - pause_ratio) * 20),
        2
    )

    label = (
        "High Confidence" if confidence_score >= 75 else
        "Moderate Confidence" if confidence_score >= 50 else
        "Low Confidence"
    )

    # -----------------------
    # RESULTS
    # -----------------------
    results = {
        "speech_rate": round(wpm),
        "pause_ratio": pause_ratio,
        "energy_level": energy_level,
        "Energy (Loudness)": round(loudness, 3),
        "Pitch Mean (semitones)": round(pitch_mean, 2),
        "Pitch Variance": round(pitch_variance, 3),
        "Jitter": round(jitter, 4),
        "Shimmer (dB)": round(shimmer, 4),
        "Speech Duration (sec)": round(duration_sec, 2),
        "Total Pause Time (sec)": total_pause_time,
        "Total Words": total_words
    }

    return results, confidence_score, label, wpm, total_pause_time

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def load_audio(path, target_sr=16000):
    """
    Loads an audio file as a mono float32 numpy array at target_sr.
    Attempts to use PyAV ('av') first to handle WebM/various formats without requiring system-wide FFmpeg.
    Falls back to librosa.load if av fails or is unavailable.
    """
    try:
        import av
        
        container = av.open(path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format='flt',  # float format
            layout='mono',
            rate=target_sr
        )
        
        audio_frames = []
        for frame in container.decode(stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    # rf.to_ndarray() has shape (1, samples) for mono, flatten it
                    audio_frames.append(rf.to_ndarray().flatten())
                    
        if audio_frames:
            audio_data = np.concatenate(audio_frames)
            logger.info(f"Successfully loaded audio using PyAV: {path} (shape: {audio_data.shape}, sr: {target_sr})")
            return audio_data, target_sr
        else:
            raise ValueError("No audio frames decoded by PyAV")
            
    except Exception as e:
        logger.warning(f"PyAV loading failed/unavailable for {path} ({e}). Falling back to librosa.load...")
        # librosa.load will use soundfile or audioread fallback
        return librosa.load(path, sr=target_sr, mono=True)

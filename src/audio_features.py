"""Audio preprocessing pipeline (spec Section 3, steps 1-2).

Step 1: resample to 22,050 Hz; extract log-mel spectrogram (128 bins) or chroma
(12 bins); normalize per track.
Step 2: split each track into fixed windows (5-10s) or beat-synchronous segments
using librosa.

Run standalone to cache features for every clip referenced in the configured
datasets:

    python -m src.audio_features --config config.yaml
"""
import argparse
import os
from pathlib import Path

import numpy as np

from src.utils import load_config, ensure_dir


def load_audio(path: str, sample_rate: int):
    """Load and resample an audio file. Requires librosa+soundfile at runtime."""
    import librosa
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y, sr


def extract_log_mel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return normalize_per_track(log_mel)


def extract_chroma(y: np.ndarray, sr: int, n_chroma: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    return normalize_per_track(chroma)


def normalize_per_track(feat: np.ndarray) -> np.ndarray:
    mu, sigma = feat.mean(), feat.std() + 1e-8
    return (feat - mu) / sigma


def segment_indices(n_frames: int, sr: int, hop_length: int, segment_seconds: float):
    """Return a list of (start_frame, end_frame) fixed-length windows covering the track."""
    frames_per_segment = int(round(segment_seconds * sr / hop_length))
    frames_per_segment = max(frames_per_segment, 1)
    bounds = []
    start = 0
    while start < n_frames:
        end = min(start + frames_per_segment, n_frames)
        if end - start > 0:
            bounds.append((start, end))
        start = end
    return bounds


def beat_synchronous_segments(y: np.ndarray, sr: int, hop_length: int):
    """Alternative to fixed windows: segment on detected beats (spec Section 3, step 2)."""
    import librosa
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_frames = np.concatenate([[0], beat_frames])
    bounds = [(int(beat_frames[i]), int(beat_frames[i + 1]))
              for i in range(len(beat_frames) - 1) if beat_frames[i + 1] > beat_frames[i]]
    return bounds if bounds else None


def segment_features(mel: np.ndarray, chroma: np.ndarray, bounds):
    """Mean-pool mel/chroma frames within each segment -> one feature vector per segment."""
    feats = []
    for (s, e) in bounds:
        mel_seg = mel[:, s:e].mean(axis=1) if e > s else mel[:, s:s + 1].mean(axis=1)
        chroma_seg = chroma[:, s:e].mean(axis=1) if e > s else chroma[:, s:s + 1].mean(axis=1)
        feats.append(np.concatenate([mel_seg, chroma_seg]))
    return np.stack(feats) if feats else np.zeros((0, mel.shape[0] + chroma.shape[0]))


def dominant_chord_per_segment(chroma: np.ndarray, bounds, chord_vocab):
    """Cheap chord estimate: argmax pitch class per segment mapped onto a
    major/minor/no-chord vocabulary via simple template correlation.
    This is a lightweight stand-in for a full chord-recognition model —
    sufficient for building a chord-transition graph as required by the spec,
    and swappable for e.g. `madmom`'s chord recognizer if higher accuracy is needed.
    """
    pitch_classes = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
    major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    chords = []
    for (s, e) in bounds:
        vec = chroma[:, s:e].mean(axis=1) if e > s else chroma[:, s]
        if vec.sum() <= 1e-6:
            chords.append("N")
            continue
        best_score, best_label = -np.inf, "N"
        for shift in range(12):
            maj = np.roll(major_template, shift)
            minr = np.roll(minor_template, shift)
            maj_score = float(np.dot(vec, maj))
            min_score = float(np.dot(vec, minr))
            if maj_score > best_score:
                best_score, best_label = maj_score, pitch_classes[shift]
            if min_score > best_score:
                best_score, best_label = min_score, pitch_classes[shift] + "m"
        chords.append(best_label if best_label in chord_vocab else "N")
    return chords


def process_clip(path: str, cfg: dict):
    a = cfg["audio"]
    y, sr = load_audio(path, a["sample_rate"])
    mel = extract_log_mel(y, sr, a["n_mels"], a["n_fft"], a["hop_length"])
    chroma = extract_chroma(y, sr, a["n_chroma"], a["n_fft"], a["hop_length"])
    n_frames = mel.shape[1]
    bounds = segment_indices(n_frames, sr, a["hop_length"], a["segment_seconds"])
    seg_feats = segment_features(mel, chroma, bounds)
    chords = dominant_chord_per_segment(chroma, bounds, cfg["graph"]["chord_vocab"])
    return {
        "mel": mel,
        "chroma": chroma,
        "segment_bounds": bounds,
        "segment_features": seg_feats,
        "segment_chords": chords,
    }


def cache_clip(clip_id: str, audio_path: str, cfg: dict, out_dir: str):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{clip_id}.npz")
    if os.path.exists(out_path):
        return out_path
    feats = process_clip(audio_path, cfg)
    np.savez_compressed(
        out_path,
        mel=feats["mel"],
        chroma=feats["chroma"],
        segment_features=feats["segment_features"],
        segment_chords=np.array(feats["segment_chords"]),
        segment_bounds=np.array(feats["segment_bounds"]),
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None, choices=["mtt", "gtzan", "deam"], help="process single dataset")
    parser.add_argument("--limit", type=int, default=None, help="cap number of clips (debug)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    from src.dataset import iter_all_clips  # local import to avoid circular import at module load

    out_dir = os.path.join(cfg["data"]["processed_dir"], "audio_features")
    n = 0
    for clip_id, audio_path, meta in iter_all_clips(cfg):
        if args.dataset and meta.get("dataset") != args.dataset:
            continue
        try:
            cache_clip(clip_id, audio_path, cfg, out_dir)
            n += 1
            if args.limit and n >= args.limit:
                break
        except Exception as e:  # keep preprocessing robust to a few corrupt files
            print(f"[audio_features] skipping {clip_id} ({audio_path}): {e}")
    print(f"[audio_features] cached {n} clips -> {out_dir}")


if __name__ == "__main__":
    main()

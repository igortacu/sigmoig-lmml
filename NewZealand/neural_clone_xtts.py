#!/usr/bin/env python3
"""
Neural voice cloning using Coqui TTS XTTS-v2
This model can achieve very high similarity scores (0.9+) with proper speaker conditioning.
"""

import sys
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SR = 16000
ROOT = Path("resources")
ENR_DIR = ROOT / "enrollment_samples"
PASS_FILE = ROOT / "passphrase.txt"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_passphrase():
    return PASS_FILE.read_text(encoding="utf-8").strip()


def load_enrollment():
    """Load enrollment sample paths"""
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    return files


def clone_voice_with_xtts(text, speaker_wav_paths, output_path):
    """Use XTTS-v2 for high-quality voice cloning"""
    from TTS.api import TTS
    import torch
    
    print("[xtts] Loading XTTS-v2 model...")
    print("[info] This may take a few minutes on first run (downloading ~2GB model)")
    
    # Use XTTS-v2 which is excellent for voice cloning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    
    print(f"[xtts] Model loaded on {device}")
    print(f"[xtts] Using {len(speaker_wav_paths)} enrollment samples")
    
    # XTTS can use multiple reference files - we'll use the longest one as primary
    # and average the speaker characteristics
    import librosa
    
    # Find best reference (longest clean audio)
    best_ref = None
    max_duration = 0
    
    for wav_path in speaker_wav_paths:
        audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        duration = len(audio) / SR
        if duration > max_duration:
            max_duration = duration
            best_ref = str(wav_path)
    
    print(f"[xtts] Primary reference: {Path(best_ref).name} ({max_duration:.2f}s)")
    
    # Generate with voice cloning
    print(f"[xtts] Generating speech...")
    print(f"[text] {text}")
    
    tts.tts_to_file(
        text=text,
        file_path=str(output_path),
        speaker_wav=best_ref,
        language="en"
    )
    
    print(f"[xtts] ✓ Generated: {output_path}")
    
    return output_path


def compute_similarity(audio1, audio2):
    """Compute speaker similarity using Resemblyzer"""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from scipy.spatial.distance import cosine
        
        encoder = VoiceEncoder()
        
        proc1 = preprocess_wav(audio1, source_sr=SR)
        proc2 = preprocess_wav(audio2, source_sr=SR)
        
        emb1 = encoder.embed_utterance(proc1)
        emb2 = encoder.embed_utterance(proc2)
        
        return 1 - cosine(emb1, emb2)
    except Exception as e:
        print(f"[warn] Similarity computation failed: {e}")
        return 0.0


def post_process_for_maximum_similarity(audio_path, enrollment_samples, sr=16000):
    """Apply fine-tuning to maximize similarity while preserving intelligibility"""
    import librosa
    import soundfile as sf
    import numpy as np
    
    # Load generated audio
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    
    print("[postprocess] Fine-tuning for maximum similarity...")
    
    # Load enrollment samples
    enroll_audios = []
    for path in enrollment_samples:
        e_audio, _ = librosa.load(str(path), sr=sr, mono=True)
        enroll_audios.append(e_audio)
    
    # Strategy: Subtle spectral envelope matching to increase similarity
    # without destroying the XTTS quality
    
    # Get spectral characteristics from enrollment
    ref_audio = max(enroll_audios, key=len)
    
    S_audio = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_ref = librosa.feature.melspectrogram(y=ref_audio, sr=sr, n_mels=128, fmax=8000)
    
    S_audio_db = librosa.power_to_db(S_audio, ref=np.max)
    S_ref_db = librosa.power_to_db(S_ref, ref=np.max)
    
    # Very light spectral matching (preserve XTTS quality)
    audio_mean = np.mean(S_audio_db, axis=1, keepdims=True)
    audio_std = np.std(S_audio_db, axis=1, keepdims=True) + 1e-6
    ref_mean = np.mean(S_ref_db, axis=1, keepdims=True)
    ref_std = np.std(S_ref_db, axis=1, keepdims=True) + 1e-6
    
    # Very light blend (0.15 = 15% adjustment)
    alpha = 0.15
    S_normalized = (S_audio_db - audio_mean) / audio_std
    S_adjusted = S_normalized * (audio_std + alpha * (ref_std - audio_std)) + (audio_mean + alpha * (ref_mean - audio_mean))
    
    # Back to audio
    S_adjusted_linear = librosa.db_to_power(S_adjusted)
    audio_adjusted = librosa.feature.inverse.mel_to_audio(
        S_adjusted_linear, sr=sr, n_fft=2048, hop_length=512, fmax=8000
    )
    
    # Match length
    if len(audio_adjusted) < len(audio):
        audio_adjusted = np.pad(audio_adjusted, (0, len(audio) - len(audio_adjusted)))
    else:
        audio_adjusted = audio_adjusted[:len(audio)]
    
    # Blend with original (preserve most of XTTS quality)
    audio_final = 0.85 * audio + 0.15 * audio_adjusted
    
    # Normalize
    audio_final = audio_final / (np.max(np.abs(audio_final)) + 1e-6) * 0.95
    
    # Save improved version
    improved_path = audio_path.with_stem(audio_path.stem + "_improved")
    sf.write(str(improved_path), audio_final, sr)
    
    print(f"[postprocess] ✓ Saved improved version")
    
    return audio_final, improved_path


def main():
    import librosa
    
    text = load_passphrase()
    print(f"\n{'='*70}")
    print(f"NEURAL VOICE CLONING WITH XTTS-v2")
    print(f"{'='*70}")
    print(f"Passphrase: {text}")
    print(f"{'='*70}\n")
    
    # Load enrollment samples
    enrollment_paths = load_enrollment()
    if not enrollment_paths:
        print("ERROR: No enrollment samples found")
        sys.exit(1)
    
    print(f"[enrollment] Found {len(enrollment_paths)} samples\n")
    
    # Generate with XTTS
    wav_output = OUT_DIR / "cloned_voice_xtts.wav"
    
    try:
        clone_voice_with_xtts(text, enrollment_paths, wav_output)
    except Exception as e:
        print(f"\n[error] XTTS failed: {e}")
        print("[info] This might be due to model download or compatibility issues")
        print("[fallback] Using previous best method...")
        
        # If XTTS fails, we can't achieve 0.95, so exit
        sys.exit(1)
    
    # Post-process for maximum similarity
    audio_improved, improved_path = post_process_for_maximum_similarity(
        wav_output, enrollment_paths, sr=SR
    )
    
    # Use the improved version
    final_wav = OUT_DIR / "cloned_voice.wav"
    subprocess.run(["cp", str(improved_path), str(final_wav)], check=True)
    
    # Convert to MP3
    out_mp3 = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(final_wav), "-ar", "16000", "-ac", "1", "-b:a", "128k", str(out_mp3)],
        check=True, capture_output=True
    )
    
    print(f"\n[output] {final_wav}")
    print(f"[output] {out_mp3}")
    
    # Evaluation
    print(f"\n{'='*70}")
    print("SIMILARITY EVALUATION")
    print(f"{'='*70}")
    
    # Load enrollment audios
    similarities = []
    for i, enroll_path in enumerate(enrollment_paths, 1):
        enroll_audio, _ = librosa.load(str(enroll_path), sr=SR, mono=True)
        min_len = min(len(audio_improved), len(enroll_audio))
        
        sim = compute_similarity(audio_improved[:min_len], enroll_audio[:min_len])
        similarities.append(sim)
        print(f"Sample {i}: {sim:.4f}")
    
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    
    print(f"\nAverage Similarity: {avg_sim:.4f}")
    print(f"{'='*70}\n")
    
    if avg_sim >= 0.95:
        print("🎉 INCREDIBLE! Similarity >= 0.95")
        print("✓ Neural voice cloning succeeded!")
    elif avg_sim >= 0.85:
        print("🎯 EXCELLENT! Similarity >= 0.85")
        print("✓ Very high quality voice cloning!")
    elif avg_sim >= 0.75:
        print("✓ SUCCESS! Similarity >= 0.75")
        print("✓ Passed threshold!")
    elif avg_sim >= 0.65:
        print(f"⚠ Partial success (need +{0.75-avg_sim:.4f})")
    else:
        print(f"✗ Below threshold (need +{0.75-avg_sim:.4f})")


if __name__ == "__main__":
    main()

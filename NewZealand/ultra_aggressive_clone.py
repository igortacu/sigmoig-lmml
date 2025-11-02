#!/usr/bin/env python3
"""
ULTRA-AGGRESSIVE voice cloning - maximum similarity optimization
Uses extreme parameter tuning and ensemble methods to push past 0.75
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
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
    import librosa
    wavs = []
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    for p in files:
        audio, _ = librosa.load(str(p), sr=SR, mono=True)
        wavs.append(audio)
    return wavs


def synth_mac(text, out_wav):
    cmd = ["say", "-o", str(out_wav), "--data-format=LEF32@16000", text]
    subprocess.run(cmd, check=True, capture_output=True)


def get_embedding(audio):
    """Get Resemblyzer embedding"""
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    processed = preprocess_wav(audio, source_sr=SR)
    return encoder.embed_utterance(processed)


def compute_similarity(audio1, audio2):
    """Compute cosine similarity"""
    from scipy.spatial.distance import cosine
    emb1 = get_embedding(audio1)
    emb2 = get_embedding(audio2)
    return 1 - cosine(emb1, emb2)


def ultra_aggressive_world_transfer(source, enrollment_samples, sr=16000):
    """EXTREME prosody and spectral transfer"""
    import pyworld as pw
    import librosa
    
    # Get ALL prosody statistics from enrollment
    all_f0, all_sp, all_ap = [], [], []
    
    for enroll in enrollment_samples:
        try:
            f0, sp, ap = pw.wav2world(enroll.astype(np.float64), sr)
            all_f0.append(f0[f0 > 0])
            all_sp.append(sp)
            all_ap.append(ap)
        except:
            continue
    
    # Aggregate target characteristics
    target_f0_mean = np.mean(np.concatenate(all_f0))
    target_f0_std = np.std(np.concatenate(all_f0))
    
    # Average spectral envelope from ALL enrollment samples
    target_sp_mean = np.mean([np.mean(sp, axis=0) for sp in all_sp], axis=0)
    target_sp_std = np.mean([np.std(sp, axis=0) for sp in all_sp], axis=0)
    
    # Extract source features
    src_f0, src_sp, src_ap = pw.wav2world(source.astype(np.float64), sr)
    
    # AGGRESSIVE F0 transformation (100% strength)
    src_f0_voiced = src_f0[src_f0 > 0]
    if len(src_f0_voiced) > 0:
        src_f0_mean = np.mean(src_f0_voiced)
        src_f0_std = np.std(src_f0_voiced)
        
        modified_f0 = src_f0.copy()
        mask = modified_f0 > 0
        
        # Full transformation
        modified_f0[mask] = ((modified_f0[mask] - src_f0_mean) / (src_f0_std + 1e-6)) * target_f0_std + target_f0_mean
        modified_f0 = np.clip(modified_f0, 50, 400)
        modified_f0[~mask] = 0
    else:
        modified_f0 = src_f0
    
    # AGGRESSIVE spectral envelope transfer (90% blend)
    src_sp_mean = np.mean(src_sp, axis=0)
    src_sp_std = np.std(src_sp, axis=0) + 1e-6
    
    modified_sp = src_sp.copy()
    blend = 0.9  # 90% target voice
    
    for i in range(len(src_sp)):
        normalized = (src_sp[i] - src_sp_mean) / src_sp_std
        transformed = normalized * target_sp_std + target_sp_mean
        modified_sp[i] = blend * transformed + (1 - blend) * src_sp[i]
    
    # Synthesize
    converted = pw.synthesize(modified_f0, modified_sp, src_ap, sr).astype(np.float32)
    converted = converted / (np.max(np.abs(converted)) + 1e-6) * 0.95
    
    return converted


def ultra_spectral_matching(source, enrollment_samples, sr=16000):
    """Maximum spectral matching to enrollment voice"""
    import librosa
    
    # Combine ALL enrollment samples for robust statistics
    all_mels = []
    for enroll in enrollment_samples:
        mel = librosa.feature.melspectrogram(y=enroll, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        all_mels.append(mel_db)
    
    # Average target spectral characteristics
    target_mel_mean = np.mean([np.mean(m, axis=1, keepdims=True) for m in all_mels], axis=0)
    target_mel_std = np.mean([np.std(m, axis=1, keepdims=True) for m in all_mels], axis=0)
    
    # Source STFT
    S_src = librosa.stft(source, n_fft=2048, hop_length=512)
    mag_src = np.abs(S_src)
    phase_src = np.angle(S_src)
    
    # Convert to mel
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
    mel_src = np.dot(mel_basis, mag_src)
    mel_src_db = librosa.power_to_db(mel_src, ref=np.max)
    
    src_mean = np.mean(mel_src_db, axis=1, keepdims=True)
    src_std = np.std(mel_src_db, axis=1, keepdims=True) + 1e-6
    
    # EXTREME transformation (85% target)
    alpha = 0.85
    mel_norm = (mel_src_db - src_mean) / src_std
    mel_matched = mel_norm * target_mel_std * alpha + target_mel_mean * alpha + mel_src_db * (1 - alpha)
    
    # Back to linear magnitude
    mel_matched_linear = librosa.db_to_power(mel_matched)
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    mag_matched = np.dot(mel_basis_pinv, mel_matched_linear)
    mag_matched = np.maximum(mag_matched, 0)
    
    # Ensure shape matches
    if mag_matched.shape[1] < mag_src.shape[1]:
        pad_width = ((0, 0), (0, mag_src.shape[1] - mag_matched.shape[1]))
        mag_matched = np.pad(mag_matched, pad_width, mode='edge')
    elif mag_matched.shape[1] > mag_src.shape[1]:
        mag_matched = mag_matched[:, :mag_src.shape[1]]
    
    # Reconstruct with original phase
    S_matched = mag_matched * np.exp(1j * phase_src)
    audio = librosa.istft(S_matched, hop_length=512)
    
    return audio


def gradient_ascent_optimization(tts, enrollment, sr=16000, iterations=10):
    """Iteratively optimize to maximize similarity"""
    import librosa
    
    print("[optimization] Gradient ascent for maximum similarity...")
    
    best_audio = tts
    best_score = 0.0
    
    # Try progressively more aggressive blends
    blend_schedule = np.linspace(0.5, 0.95, iterations)
    
    for i, blend in enumerate(blend_schedule, 1):
        print(f"[iteration {i}/{iterations}] Blend: {blend:.3f}", end=" ")
        
        # Apply WORLD transfer
        conv = ultra_aggressive_world_transfer(tts, enrollment, sr)
        
        # Apply spectral matching with current blend
        temp_audio = conv.copy()
        for enroll in enrollment[:3]:  # Use top 3 samples
            temp = ultra_spectral_matching(temp_audio, [enroll], sr)
            # Match lengths before blending
            min_len = min(len(temp), len(temp_audio))
            temp = temp[:min_len]
            temp_audio_trimmed = temp_audio[:min_len]
            blended = blend * temp + (1 - blend) * temp_audio_trimmed
            temp_audio = blended
        
        # Evaluate
        sims = []
        for enroll in enrollment:
            min_len = min(len(temp_audio), len(enroll))
            sim = compute_similarity(temp_audio[:min_len], enroll[:min_len])
            sims.append(sim)
        
        avg_sim = np.mean(sims)
        print(f"→ {avg_sim:.4f}")
        
        if avg_sim > best_score:
            best_score = avg_sim
            best_audio = temp_audio
            print(f"           ✓ NEW BEST")
    
    return best_audio, best_score


def ensemble_approach(tts, enrollment, sr=16000):
    """Generate multiple variants and ensemble best"""
    import librosa
    
    print("\n[ensemble] Generating multiple conversion variants...")
    
    variants = []
    
    # Variant 1: Pure WORLD extreme
    v1 = ultra_aggressive_world_transfer(tts, enrollment, sr)
    variants.append(("WORLD-100%", v1))
    
    # Variant 2: WORLD + spectral
    v2 = ultra_aggressive_world_transfer(tts, enrollment, sr)
    v2 = ultra_spectral_matching(v2, enrollment, sr)
    variants.append(("WORLD+Spectral", v2))
    
    # Variant 3: Double pass WORLD
    v3 = ultra_aggressive_world_transfer(tts, enrollment, sr)
    v3 = ultra_aggressive_world_transfer(v3, enrollment, sr)  # Second pass
    variants.append(("WORLD-Double", v3))
    
    # Variant 4: Spectral-first
    v4 = ultra_spectral_matching(tts, enrollment, sr)
    v4 = ultra_aggressive_world_transfer(v4, enrollment, sr)
    variants.append(("Spectral-First", v4))
    
    # Evaluate all
    best_score = 0
    best_variant = None
    best_name = ""
    
    for name, audio in variants:
        sims = []
        for enroll in enrollment:
            min_len = min(len(audio), len(enroll))
            sim = compute_similarity(audio[:min_len], enroll[:min_len])
            sims.append(sim)
        
        avg_sim = np.mean(sims)
        print(f"  {name:20s}: {avg_sim:.4f}")
        
        if avg_sim > best_score:
            best_score = avg_sim
            best_variant = audio
            best_name = name
    
    print(f"\n[selected] {best_name} (score: {best_score:.4f})")
    
    return best_variant, best_score


def main():
    import librosa, soundfile as sf
    
    print(f"\n{'='*70}")
    print("ULTRA-AGGRESSIVE VOICE CLONING")
    print("Target: 0.95 similarity (extreme optimization)")
    print(f"{'='*70}\n")
    
    text = load_passphrase()
    print(f"Passphrase: {text}\n")
    
    enrollment = load_enrollment()
    if not enrollment:
        print("ERROR: No enrollment samples")
        sys.exit(1)
    
    print(f"[enrollment] {len(enrollment)} samples\n")
    
    # Generate TTS
    tts_path = OUT_DIR / "tts_raw.wav"
    print("[tts] Generating base...")
    synth_mac(text, tts_path)
    tts, _ = librosa.load(str(tts_path), sr=SR, mono=True)
    print(f"[tts] {len(tts)/SR:.2f}s\n")
    
    # Ensemble approach
    converted, ensemble_score = ensemble_approach(tts, enrollment, SR)
    
    print(f"\n[ensemble] Best score: {ensemble_score:.4f}")
    
    # Further optimize with gradient ascent
    final, final_score = gradient_ascent_optimization(converted, enrollment, SR, iterations=15)
    
    # Save
    out_wav = OUT_DIR / "cloned_voice.wav"
    sf.write(str(out_wav), final, SR)
    
    out_mp3 = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav), "-ar", "16000", "-ac", "1", "-b:a", "128k", str(out_mp3)],
        check=True, capture_output=True
    )
    
    print(f"\n[output] {out_wav}")
    print(f"[output] {out_mp3}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    for i, enroll in enumerate(enrollment, 1):
        min_len = min(len(final), len(enroll))
        sim = compute_similarity(final[:min_len], enroll[:min_len])
        print(f"Sample {i}: {sim:.4f}")
    
    avg = np.mean([compute_similarity(final[:min(len(final), len(e))], e[:min(len(final), len(e))]) 
                   for e in enrollment])
    
    print(f"\nAverage: {avg:.4f}")
    print(f"{'='*70}\n")
    
    if avg >= 0.95:
        print("🎉🎉🎉 INCREDIBLE! >= 0.95!")
    elif avg >= 0.85:
        print("🎯 EXCELLENT! >= 0.85!")
    elif avg >= 0.75:
        print("✓ SUCCESS! >= 0.75!")
    else:
        print(f"⚠ Need +{0.75-avg:.4f}")
        print("Note: 0.95 is extremely difficult without neural TTS")


if __name__ == "__main__":
    main()

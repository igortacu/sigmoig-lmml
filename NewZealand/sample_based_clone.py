#!/usr/bin/env python3
"""
Sample-based voice cloning using enrollment audio as base material.
Uses concatenative synthesis and prosody transfer from enrollment samples.
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


def load_passphrase() -> str:
    return PASS_FILE.read_text(encoding="utf-8").strip()


def load_enrollment():
    """Load enrollment samples"""
    import librosa
    wavs = []
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    for p in files:
        audio, _ = librosa.load(str(p), sr=SR, mono=True)
        wavs.append(audio)
    return wavs


def synth_mac(text: str, out_wav: Path):
    """Generate TTS using macOS"""
    cmd = ["say", "-o", str(out_wav), "--data-format=LEF32@16000", text]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_prosody_from_enrollment(enrollment_samples, sr=16000):
    """Extract average prosody characteristics from enrollment samples"""
    import librosa
    import parselmouth
    
    all_f0 = []
    all_energy = []
    all_duration = []
    all_spectral_centroid = []
    
    for audio in enrollment_samples:
        # F0
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = snd.to_pitch(time_step=0.01)
            f0 = pitch.selected_array['frequency']
            f0_voiced = f0[f0 > 0]
            all_f0.extend(f0_voiced)
        except:
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
            f0_voiced = f0[f0 > 0]
            all_f0.extend(f0_voiced)
        
        # Energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        all_energy.extend(rms)
        
        # Duration
        all_duration.append(len(audio) / sr)
        
        # Spectral characteristics
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        all_spectral_centroid.extend(spec_cent)
    
    return {
        'f0_mean': np.mean(all_f0) if len(all_f0) > 0 else 150.0,
        'f0_std': np.std(all_f0) if len(all_f0) > 0 else 20.0,
        'f0_median': np.median(all_f0) if len(all_f0) > 0 else 150.0,
        'f0_min': np.percentile(all_f0, 5) if len(all_f0) > 0 else 80.0,
        'f0_max': np.percentile(all_f0, 95) if len(all_f0) > 0 else 250.0,
        'energy_mean': np.mean(all_energy),
        'energy_std': np.std(all_energy),
        'avg_duration': np.mean(all_duration),
        'spectral_centroid': np.mean(all_spectral_centroid)
    }


def transfer_voice_characteristics_advanced(source_audio, target_prosody, enrollment_samples, sr=16000):
    """Advanced voice transfer using WORLD vocoder and spectral matching"""
    import pyworld as pw
    import librosa
    
    # Extract WORLD features from source
    source_f0, source_sp, source_ap = pw.wav2world(source_audio.astype(np.float64), sr)
    
    # Prosody transfer - match F0 carefully
    source_f0_voiced = source_f0[source_f0 > 0]
    if len(source_f0_voiced) > 0:
        source_f0_mean = np.mean(source_f0_voiced)
        source_f0_std = np.std(source_f0_voiced)
        
        # Target prosody parameters
        target_f0_mean = target_prosody['f0_mean']
        target_f0_std = target_prosody['f0_std']
        
        # Transform F0 contour
        modified_f0 = source_f0.copy()
        voiced_mask = modified_f0 > 0
        
        # Normalize and rescale
        modified_f0[voiced_mask] = ((modified_f0[voiced_mask] - source_f0_mean) / (source_f0_std + 1e-6)) * target_f0_std + target_f0_mean
        
        # Clip to reasonable range
        modified_f0 = np.clip(modified_f0, 50, 400)
        modified_f0[~voiced_mask] = 0
    else:
        modified_f0 = source_f0
    
    # Spectral envelope adaptation
    # Extract spectral statistics from enrollment samples
    enrollment_sp_stats = []
    for enroll in enrollment_samples:
        try:
            enroll_f0, enroll_sp, enroll_ap = pw.wav2world(enroll.astype(np.float64), sr)
            sp_mean = np.mean(enroll_sp, axis=0)
            sp_std = np.std(enroll_sp, axis=0)
            enrollment_sp_stats.append({'mean': sp_mean, 'std': sp_std})
        except:
            continue
    
    if enrollment_sp_stats:
        # Average enrollment spectral characteristics
        target_sp_mean = np.mean([s['mean'] for s in enrollment_sp_stats], axis=0)
        target_sp_std = np.mean([s['std'] for s in enrollment_sp_stats], axis=0)
        
        # Source spectral statistics
        source_sp_mean = np.mean(source_sp, axis=0)
        source_sp_std = np.std(source_sp, axis=0) + 1e-6
        
        # Apply spectral transformation (blend to avoid over-processing)
        modified_sp = source_sp.copy()
        blend_factor = 0.6
        
        for i in range(len(modified_sp)):
            normalized_sp = (source_sp[i] - source_sp_mean) / source_sp_std
            transformed_sp = normalized_sp * target_sp_std + target_sp_mean
            modified_sp[i] = blend_factor * transformed_sp + (1 - blend_factor) * source_sp[i]
    else:
        modified_sp = source_sp
    
    # Synthesize with WORLD
    converted = pw.synthesize(modified_f0, modified_sp, source_ap, sr).astype(np.float32)
    
    # Post-processing: match energy envelope to target
    target_energy = target_prosody['energy_mean']
    current_energy = np.sqrt(np.mean(converted ** 2))
    if current_energy > 0:
        energy_ratio = target_energy / current_energy
        converted = converted * energy_ratio * 0.9  # 0.9 to avoid clipping
    
    # Normalize
    converted = converted / (np.max(np.abs(converted)) + 1e-6) * 0.95
    
    return converted


def get_speaker_embedding_resemblyzer(audio):
    """Extract speaker embedding using Resemblyzer"""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        processed = preprocess_wav(audio, source_sr=SR)
        return encoder.embed_utterance(processed)
    except Exception as e:
        return None


def compute_similarity(audio1, audio2):
    """Compute speaker similarity using Resemblyzer"""
    from scipy.spatial.distance import cosine
    emb1 = get_speaker_embedding_resemblyzer(audio1)
    emb2 = get_speaker_embedding_resemblyzer(audio2)
    if emb1 is None or emb2 is None:
        return 0.0
    return 1 - cosine(emb1, emb2)


def iterative_refinement(tts_audio, enrollment_samples, target_prosody, sr=16000, max_iterations=5):
    """Iteratively refine voice conversion to maximize similarity"""
    import librosa
    
    print("[refinement] Starting iterative optimization...")
    
    best_audio = tts_audio
    best_score = 0.0
    
    # Try different blending strengths and parameters
    blend_factors = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    for iteration, blend in enumerate(blend_factors, 1):
        print(f"[iteration {iteration}] Testing blend factor: {blend:.2f}")
        
        # Apply conversion with current parameters
        converted = transfer_voice_characteristics_advanced(
            tts_audio, target_prosody, enrollment_samples, sr
        )
        
        # Apply additional spectral matching with varying strength
        ref_audio = max(enrollment_samples, key=len)
        
        S_converted = librosa.stft(converted, n_fft=2048, hop_length=512)
        S_ref = librosa.stft(ref_audio, n_fft=2048, hop_length=512)
        
        mag_converted = np.abs(S_converted)
        mag_ref = np.abs(S_ref)
        phase_converted = np.angle(S_converted)
        
        # Match spectral envelope
        mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80)
        mel_converted = np.dot(mel_basis, mag_converted)
        mel_ref = np.dot(mel_basis, mag_ref)
        
        mel_conv_mean = np.mean(mel_converted, axis=1, keepdims=True)
        mel_conv_std = np.std(mel_converted, axis=1, keepdims=True) + 1e-6
        mel_ref_mean = np.mean(mel_ref, axis=1, keepdims=True)
        mel_ref_std = np.std(mel_ref, axis=1, keepdims=True) + 1e-6
        
        mel_normalized = (mel_converted - mel_conv_mean) / mel_conv_std
        mel_transformed = mel_normalized * mel_ref_std * blend + mel_converted * (1 - blend) + mel_ref_mean * blend
        
        # Back to linear
        mel_basis_pinv = np.linalg.pinv(mel_basis)
        mag_final = np.dot(mel_basis_pinv, mel_transformed)
        mag_final = np.maximum(mag_final, 0)
        
        # Ensure shape compatibility
        if mag_final.shape[1] < mag_converted.shape[1]:
            mag_final = np.pad(mag_final, ((0, 0), (0, mag_converted.shape[1] - mag_final.shape[1])), mode='edge')
        elif mag_final.shape[1] > mag_converted.shape[1]:
            mag_final = mag_final[:, :mag_converted.shape[1]]
        
        S_final = mag_final * np.exp(1j * phase_converted)
        refined = librosa.istft(S_final, hop_length=512)
        
        # Normalize
        refined = refined / (np.max(np.abs(refined)) + 1e-6) * 0.95
        
        # Evaluate similarity
        similarities = []
        for enroll in enrollment_samples:
            min_len = min(len(refined), len(enroll))
            sim = compute_similarity(refined[:min_len], enroll[:min_len])
            similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        print(f"  Similarity: {avg_sim:.4f}")
        
        if avg_sim > best_score:
            best_score = avg_sim
            best_audio = refined
            print(f"  ✓ New best score!")
    
    print(f"\n[result] Best similarity achieved: {best_score:.4f}")
    return best_audio, best_score


def main():
    import librosa
    import soundfile as sf
    
    text = load_passphrase()
    print(f"[passphrase] {text}\n")
    
    # Load enrollment samples
    enrollment_samples = load_enrollment()
    if not enrollment_samples:
        print("ERROR: No enrollment samples found")
        sys.exit(1)
    
    print(f"[enrollment] Loaded {len(enrollment_samples)} samples\n")
    
    # Extract target prosody from enrollment
    print("[analysis] Extracting speaker characteristics...")
    target_prosody = extract_prosody_from_enrollment(enrollment_samples, sr=SR)
    print(f"[prosody] F0: {target_prosody['f0_mean']:.1f} Hz (±{target_prosody['f0_std']:.1f})")
    print(f"[prosody] Range: {target_prosody['f0_min']:.1f} - {target_prosody['f0_max']:.1f} Hz\n")
    
    # Generate base TTS
    tts_raw = OUT_DIR / "tts_raw.wav"
    print("[tts] Generating speech with macOS voice...")
    try:
        synth_mac(text, tts_raw)
    except Exception as e:
        print(f"ERROR: TTS failed: {e}")
        sys.exit(1)
    
    tts_audio, _ = librosa.load(str(tts_raw), sr=SR, mono=True)
    print(f"[tts] Duration: {len(tts_audio)/SR:.2f}s\n")
    
    # Apply iterative refinement for best results
    converted, final_score = iterative_refinement(
        tts_audio, enrollment_samples, target_prosody, sr=SR
    )
    
    # Save outputs
    out_wav = OUT_DIR / "cloned_voice.wav"
    sf.write(str(out_wav), converted, SR)
    print(f"\n[output] {out_wav}")
    
    # Convert to MP3
    out_mp3 = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav), "-ar", "16000", "-ac", "1", "-b:a", "128k", str(out_mp3)],
        check=True, capture_output=True
    )
    print(f"[output] {out_mp3}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION")
    print(f"{'='*60}")
    
    for i, ref_audio in enumerate(enrollment_samples, 1):
        min_len = min(len(converted), len(ref_audio))
        sim = compute_similarity(converted[:min_len], ref_audio[:min_len])
        print(f"Sample {i}: {sim:.4f}")
    
    print(f"\nAverage Similarity: {final_score:.4f}")
    print(f"{'='*60}\n")
    
    if final_score >= 0.75:
        print("✓ SUCCESS! Similarity >= 0.75 threshold")
        print("🎉 Voice cloning complete!")
    elif final_score >= 0.65:
        print(f"⚠ Partial success. Need +{0.75 - final_score:.4f} to reach threshold")
    else:
        print(f"✗ Below passing threshold. Need +{0.75 - final_score:.4f}")


if __name__ == "__main__":
    main()

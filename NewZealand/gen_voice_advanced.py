#!/usr/bin/env python3
"""
Advanced voice cloning using feature extraction from enrollment samples
and transfer to TTS output.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d

SR = 16000
ROOT = Path("resources")
ENR_DIR = ROOT / "enrollment_samples"
PASS_FILE = ROOT / "passphrase.txt"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import parselmouth
    HAS_PRAAT = True
except ImportError:
    HAS_PRAAT = False
    print("[warning] praat-parselmouth not available")

try:
    import pyworld as pw
    HAS_WORLD = True
except ImportError:
    HAS_WORLD = False
    print("[warning] pyworld not available")


def load_passphrase() -> str:
    return PASS_FILE.read_text(encoding="utf-8").strip()


def synth_tts(text: str, out_wav: Path):
    """Generate TTS using macOS voice"""
    cmd = ["say", "-o", str(out_wav), "--data-format=LEF32@16000", text]
    subprocess.run(cmd, check=True)


def load_enrollment():
    """Load all enrollment samples"""
    wavs = []
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    for p in files:
        audio, _ = librosa.load(str(p), sr=SR, mono=True)
        wavs.append(audio)
    return wavs


def extract_f0_contour(audio, sr=SR):
    """Extract F0 contour using multiple methods for robustness"""
    if HAS_WORLD:
        # WORLD vocoder - more accurate
        audio_double = audio.astype(np.float64)
        f0, timeaxis = pw.dio(audio_double, sr, frame_period=10)
        f0 = pw.stonemask(audio_double, f0, timeaxis, sr)
    else:
        # Fallback to librosa
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr, frame_length=2048)
        timeaxis = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    
    return f0, timeaxis


def extract_spectral_envelope(audio, sr=SR):
    """Extract spectral envelope characteristics"""
    if HAS_WORLD:
        audio_double = audio.astype(np.float64)
        f0, timeaxis = pw.dio(audio_double, sr, frame_period=10)
        f0 = pw.stonemask(audio_double, f0, timeaxis, sr)
        sp = pw.cheaptrick(audio_double, f0, timeaxis, sr)
        ap = pw.d4c(audio_double, f0, timeaxis, sr)
        return sp, ap, f0, timeaxis
    return None, None, None, None


def analyze_enrollment_features(enrollment_wavs):
    """Extract comprehensive features from enrollment samples"""
    features = {
        'f0_mean': [],
        'f0_std': [],
        'f0_range': [],
        'spectral_centroids': [],
        'spectral_rolloffs': [],
        'formants': [],
        'energy_profiles': [],
        'speaking_rates': [],
    }
    
    for wav in enrollment_wavs:
        # F0 analysis
        f0, _ = extract_f0_contour(wav)
        f0_voiced = f0[f0 > 0]
        if len(f0_voiced) > 0:
            features['f0_mean'].append(np.mean(f0_voiced))
            features['f0_std'].append(np.std(f0_voiced))
            features['f0_range'].append((np.percentile(f0_voiced, 5), np.percentile(f0_voiced, 95)))
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=wav, sr=SR)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=wav, sr=SR, roll_percent=0.85)[0]
        features['spectral_centroids'].append(np.mean(spec_cent))
        features['spectral_rolloffs'].append(np.mean(spec_roll))
        
        # Energy profile
        rms = librosa.feature.rms(y=wav)[0]
        features['energy_profiles'].append(rms)
        
        # Extract formants using Praat if available
        if HAS_PRAAT:
            try:
                snd = parselmouth.Sound(wav, sampling_frequency=SR)
                formants = snd.to_formant_burg()
                f1_vals, f2_vals, f3_vals = [], [], []
                for t in np.linspace(0, snd.duration, 20):
                    f1 = formants.get_value_at_time(1, t)
                    f2 = formants.get_value_at_time(2, t)
                    f3 = formants.get_value_at_time(3, t)
                    if not np.isnan(f1):
                        f1_vals.append(f1)
                    if not np.isnan(f2):
                        f2_vals.append(f2)
                    if not np.isnan(f3):
                        f3_vals.append(f3)
                
                if f1_vals and f2_vals:
                    features['formants'].append({
                        'f1': np.mean(f1_vals),
                        'f2': np.mean(f2_vals),
                        'f3': np.mean(f3_vals) if f3_vals else None
                    })
            except Exception as e:
                print(f"[formant extraction failed: {e}]")
    
    # Compute average features
    target_features = {
        'f0_mean': np.mean(features['f0_mean']) if features['f0_mean'] else 150.0,
        'f0_std': np.mean(features['f0_std']) if features['f0_std'] else 20.0,
        'f0_range': (
            np.mean([r[0] for r in features['f0_range']]) if features['f0_range'] else 100.0,
            np.mean([r[1] for r in features['f0_range']]) if features['f0_range'] else 200.0
        ),
        'spectral_centroid': np.mean(features['spectral_centroids']),
        'spectral_rolloff': np.mean(features['spectral_rolloffs']),
    }
    
    if features['formants']:
        target_features['formants'] = {
            'f1': np.mean([f['f1'] for f in features['formants']]),
            'f2': np.mean([f['f2'] for f in features['formants']]),
            'f3': np.mean([f['f3'] for f in features['formants'] if f['f3']]) if any(f['f3'] for f in features['formants']) else None
        }
    
    return target_features


def convert_voice_world(audio, target_features, sr=SR):
    """Convert voice using WORLD vocoder for high-quality conversion"""
    if not HAS_WORLD:
        print("[warning] WORLD not available, using simple method")
        return convert_voice_simple(audio, target_features, sr)
    
    audio_double = audio.astype(np.float64)
    
    # Extract source features
    f0_src, timeaxis = pw.dio(audio_double, sr, frame_period=5)
    f0_src = pw.stonemask(audio_double, f0_src, timeaxis, sr)
    sp_src = pw.cheaptrick(audio_double, f0_src, timeaxis, sr)
    ap_src = pw.d4c(audio_double, f0_src, timeaxis, sr)
    
    # Modify F0 to match target
    f0_tgt = np.copy(f0_src)
    voiced_indices = f0_src > 0
    
    if np.any(voiced_indices):
        src_f0_mean = np.mean(f0_src[voiced_indices])
        target_f0_mean = target_features['f0_mean']
        
        # Scale F0 to match target mean while preserving contour
        f0_ratio = target_f0_mean / src_f0_mean
        f0_tgt[voiced_indices] = f0_src[voiced_indices] * f0_ratio
        
        # Adjust F0 variance to match target
        target_f0_std = target_features['f0_std']
        src_f0_std = np.std(f0_src[voiced_indices])
        if src_f0_std > 0:
            f0_tgt_centered = f0_tgt[voiced_indices] - target_f0_mean
            f0_tgt_scaled = f0_tgt_centered * (target_f0_std / src_f0_std)
            f0_tgt[voiced_indices] = f0_tgt_scaled + target_f0_mean
        
        # Clip to target range
        f0_min, f0_max = target_features['f0_range']
        f0_tgt[voiced_indices] = np.clip(f0_tgt[voiced_indices], f0_min, f0_max)
    
    # Modify spectral envelope
    sp_tgt = np.copy(sp_src)
    
    # Spectral tilt adjustment
    freq_bins = sp_tgt.shape[1]
    freqs = np.linspace(0, sr/2, freq_bins)
    
    # Compute current and target spectral characteristics
    src_spectral_mean = np.mean(sp_src, axis=0)
    
    # Formant shifting if available
    if 'formants' in target_features and HAS_PRAAT:
        try:
            # Create a formant filter to shift peaks
            f1_target = target_features['formants']['f1']
            f2_target = target_features['formants']['f2']
            
            # Simple spectral warping toward target formants
            # This is a simplified approach - full formant synthesis would be better
            formant_scale = (f1_target / 700.0)  # 700 Hz typical F1
            
            # Apply gentle spectral warping
            for i in range(sp_tgt.shape[0]):
                # Interpolate to warp frequency axis
                interp_func = interp1d(freqs, sp_tgt[i], kind='cubic', fill_value='extrapolate')
                warped_freqs = freqs * formant_scale
                warped_freqs = np.clip(warped_freqs, 0, sr/2)
                sp_tgt[i] = interp_func(warped_freqs)
        except Exception as e:
            print(f"[formant shifting failed: {e}]")
    
    # Spectral envelope scaling for timbre matching
    target_centroid = target_features['spectral_centroid']
    src_centroid = np.sum(freqs * src_spectral_mean) / np.sum(src_spectral_mean)
    
    if src_centroid > 0:
        # Create a tilt filter
        centroid_ratio = target_centroid / src_centroid
        tilt = np.power(freqs / (sr/2) + 0.1, (centroid_ratio - 1.0) * 0.3)
        tilt = tilt.reshape(1, -1)
        sp_tgt = sp_tgt * tilt
    
    # Synthesize with modified parameters
    try:
        converted = pw.synthesize(f0_tgt, sp_tgt, ap_src, sr, frame_period=5)
        converted = converted.astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(converted))
        if max_val > 0:
            converted = converted / max_val * 0.95
        
        return converted
    except Exception as e:
        print(f"[WORLD synthesis failed: {e}]")
        return convert_voice_simple(audio, target_features, sr)


def convert_voice_simple(audio, target_features, sr=SR):
    """Simpler voice conversion fallback"""
    # Extract source F0
    f0_src, _ = extract_f0_contour(audio, sr)
    
    # Apply pitch shift
    voiced = f0_src > 0
    if np.any(voiced):
        src_mean = np.mean(f0_src[voiced])
        target_mean = target_features['f0_mean']
        
        semitones = 12 * np.log2(target_mean / src_mean)
        semitones = np.clip(semitones, -12, 12)
        
        print(f"[simple] pitch shift: {semitones:.2f} semitones")
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    
    # Spectral envelope adjustment
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag, phase = np.abs(D), np.angle(D)
    
    # Adjust spectral centroid
    n_bins = mag.shape[0]
    target_centroid = target_features['spectral_centroid']
    
    # Create frequency-dependent gain
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    current_energy = np.sum(mag, axis=1)
    current_centroid = np.sum(freqs * current_energy) / (np.sum(current_energy) + 1e-8)
    
    # Tilt toward target centroid
    if current_centroid > 0:
        ratio = target_centroid / current_centroid
        tilt_factor = np.power(freqs / (sr/2) + 0.1, (ratio - 1.0) * 0.5)
        mag = mag * tilt_factor.reshape(-1, 1)
    
    # Reconstruct
    D_modified = mag * np.exp(1j * phase)
    audio = librosa.istft(D_modified, hop_length=512)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    return audio


def main():
    text = load_passphrase()
    print(f"[passphrase] {text}")
    
    # Load enrollment samples
    enrollment = load_enrollment()
    if not enrollment:
        print("[error] No enrollment samples found")
        sys.exit(1)
    
    print(f"[enrollment] Loaded {len(enrollment)} samples")
    
    # Extract target features from enrollment
    print("[analysis] Extracting target speaker features...")
    target_features = analyze_enrollment_features(enrollment)
    
    print(f"[target] F0 mean: {target_features['f0_mean']:.1f} Hz")
    print(f"[target] F0 std: {target_features['f0_std']:.1f} Hz")
    print(f"[target] F0 range: {target_features['f0_range'][0]:.1f} - {target_features['f0_range'][1]:.1f} Hz")
    print(f"[target] Spectral centroid: {target_features['spectral_centroid']:.1f} Hz")
    if 'formants' in target_features:
        print(f"[target] Formants: F1={target_features['formants']['f1']:.0f} Hz, F2={target_features['formants']['f2']:.0f} Hz")
    
    # Generate TTS
    tts_wav = OUT_DIR / "tts_raw.wav"
    print("[tts] Generating base speech...")
    try:
        synth_tts(text, tts_wav)
    except Exception as e:
        print(f"[error] TTS failed: {e}")
        sys.exit(1)
    
    # Load TTS audio
    tts_audio, _ = librosa.load(str(tts_wav), sr=SR, mono=True)
    print(f"[tts] Duration: {len(tts_audio)/SR:.2f}s")
    
    # Apply voice conversion
    print("[conversion] Applying voice conversion...")
    converted = convert_voice_world(tts_audio, target_features, SR)
    
    # Ensure proper length
    if len(converted) > len(tts_audio) * 1.2:
        converted = converted[:len(tts_audio)]
    
    # Save WAV
    out_wav = OUT_DIR / "cloned_voice.wav"
    sf.write(str(out_wav), converted, SR)
    print(f"[output] {out_wav}")
    
    # Convert to MP3
    out_mp3 = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav), "-ar", "16000", "-ac", "1", "-b:a", "64k", str(out_mp3)],
        check=True,
        capture_output=True
    )
    print(f"[output] {out_mp3}")
    
    # Evaluate with resemblyzer if available
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        
        print("\n[evaluation] Computing speaker similarity...")
        converted_embed = encoder.embed_utterance(preprocess_wav(converted, SR))
        
        similarities = []
        for i, enr_wav in enumerate(enrollment, 1):
            enr_embed = encoder.embed_utterance(preprocess_wav(enr_wav, SR))
            sim = np.dot(converted_embed, enr_embed) / (np.linalg.norm(converted_embed) * np.linalg.norm(enr_embed))
            similarities.append(sim)
            print(f"  Sample {i}: {sim:.4f}")
        
        avg_sim = np.mean(similarities)
        print(f"\n[result] Average similarity: {avg_sim:.4f}")
        if avg_sim >= 0.75:
            print("[SUCCESS] Similarity >= 0.75 threshold! 🎉")
        else:
            print(f"[INFO] Need {0.75 - avg_sim:.4f} more to reach threshold")
            
    except ImportError:
        print("\n[note] Install resemblyzer to evaluate similarity")
    except Exception as e:
        print(f"\n[evaluation error] {e}")


if __name__ == "__main__":
    main()

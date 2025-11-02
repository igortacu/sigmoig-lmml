#!/usr/bin/env python3
"""
Ultimate voice cloning with multi-strategy optimization and similarity-guided refinement.
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


def synth_mac_voice(text: str, out_wav: Path, voice_name=None):
    """Generate TTS using macOS with specific voice"""
    cmd = ["say", "-o", str(out_wav), "--data-format=LEF32@16000"]
    if voice_name:
        cmd.extend(["-v", voice_name])
    cmd.append(text)
    subprocess.run(cmd, check=True, capture_output=True)


def get_available_voices():
    """Get list of available macOS voices"""
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        voices = []
        for line in result.stdout.split('\n'):
            if line.strip():
                voice_name = line.split()[0]
                voices.append(voice_name)
        return voices
    except:
        return ["Samantha", "Alex", "Victoria", "Karen", "Moira"]


def load_enrollment():
    """Load enrollment samples"""
    import librosa
    wavs = []
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    for p in files:
        audio, _ = librosa.load(str(p), sr=SR, mono=True)
        wavs.append(audio)
    return wavs


def get_speaker_embedding(audio, method='resemblyzer'):
    """Extract speaker embedding using specified method"""
    try:
        if method == 'resemblyzer':
            from resemblyzer import VoiceEncoder, preprocess_wav
            encoder = VoiceEncoder()
            processed = preprocess_wav(audio, source_sr=SR)
            return encoder.embed_utterance(processed)
        elif method == 'speechbrain':
            from speechbrain.pretrained import EncoderClassifier
            import torch
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            embedding = classifier.encode_batch(audio_tensor)
            return embedding.squeeze().numpy()
    except Exception as e:
        print(f"[warn] {method} failed: {e}")
        return None


def compute_similarity(audio1, audio2, method='resemblyzer'):
    """Compute speaker similarity"""
    from scipy.spatial.distance import cosine
    emb1 = get_speaker_embedding(audio1, method)
    emb2 = get_speaker_embedding(audio2, method)
    if emb1 is None or emb2 is None:
        return 0.0
    return 1 - cosine(emb1, emb2)


def extract_detailed_prosody(audio, sr=16000):
    """Extract comprehensive prosody features"""
    import librosa
    import parselmouth
    
    # F0 contour with Praat
    try:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01)
        f0_contour = pitch.selected_array['frequency']
        times = pitch.xs()
        
        # Get statistics
        f0_voiced = f0_contour[f0_contour > 0]
        f0_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 150.0
        f0_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 20.0
        f0_median = np.median(f0_voiced) if len(f0_voiced) > 0 else 150.0
        
        # Pitch range
        f0_min = np.percentile(f0_voiced, 5) if len(f0_voiced) > 0 else 80.0
        f0_max = np.percentile(f0_voiced, 95) if len(f0_voiced) > 0 else 250.0
        
    except:
        f0_contour = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
        f0_voiced = f0_contour[f0_contour > 0]
        f0_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 150.0
        f0_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 20.0
        f0_median = np.median(f0_voiced) if len(f0_voiced) > 0 else 150.0
        f0_min = np.percentile(f0_voiced, 5) if len(f0_voiced) > 0 else 80.0
        f0_max = np.percentile(f0_voiced, 95) if len(f0_voiced) > 0 else 250.0
    
    # Energy and dynamics
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    
    # Spectral characteristics
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    return {
        'f0_mean': f0_mean,
        'f0_median': f0_median,
        'f0_std': f0_std,
        'f0_min': f0_min,
        'f0_max': f0_max,
        'f0_range': f0_max - f0_min,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'spec_centroid_mean': np.mean(spec_centroid),
        'spec_rolloff_mean': np.mean(spec_rolloff)
    }


def advanced_prosody_transfer(source_audio, target_prosody, sr=16000, strength=0.8):
    """Advanced prosody transfer with WORLD vocoder"""
    import pyworld as pw
    import librosa
    
    # Extract source features
    source_f0, source_sp, source_ap = pw.wav2world(
        source_audio.astype(np.float64), sr
    )
    
    # Get source prosody stats
    source_f0_voiced = source_f0[source_f0 > 0]
    if len(source_f0_voiced) == 0:
        return source_audio
    
    source_f0_mean = np.mean(source_f0_voiced)
    source_f0_std = np.std(source_f0_voiced)
    
    # Calculate transformation
    # Shift mean and scale range
    f0_shift = (target_prosody['f0_mean'] - source_f0_mean) * strength
    f0_scale = (target_prosody['f0_std'] / (source_f0_std + 1e-6)) ** strength
    
    # Apply transformation
    modified_f0 = source_f0.copy()
    voiced_mask = modified_f0 > 0
    modified_f0[voiced_mask] = (modified_f0[voiced_mask] - source_f0_mean) * f0_scale + target_prosody['f0_mean']
    
    # Clip to reasonable range
    modified_f0 = np.clip(modified_f0, 50, 400)
    modified_f0[~voiced_mask] = 0
    
    # Formant shifting via spectral envelope warping
    # Lower frequencies for female-to-male, higher for male-to-female
    freq_ratio = target_prosody['f0_mean'] / source_f0_mean
    
    # Apply spectral warping
    modified_sp = source_sp.copy()
    if freq_ratio < 0.9:  # Lower voice
        # Compress spectrum (lower formants)
        for i in range(len(modified_sp)):
            modified_sp[i] = np.interp(
                np.linspace(0, len(source_sp[i])-1, len(source_sp[i])),
                np.linspace(0, len(source_sp[i])-1, int(len(source_sp[i])/freq_ratio)),
                np.pad(source_sp[i], (0, int(len(source_sp[i])*(1/freq_ratio-1))), mode='edge')[:int(len(source_sp[i])/freq_ratio)]
            )
    elif freq_ratio > 1.1:  # Higher voice
        # Expand spectrum (raise formants)
        for i in range(len(modified_sp)):
            expanded = np.interp(
                np.linspace(0, len(source_sp[i])-1, int(len(source_sp[i])*freq_ratio)),
                np.linspace(0, len(source_sp[i])-1, len(source_sp[i])),
                source_sp[i]
            )
            modified_sp[i] = expanded[:len(source_sp[i])]
    
    # Synthesize
    converted = pw.synthesize(
        modified_f0, modified_sp, source_ap, sr
    ).astype(np.float32)
    
    # Normalize
    converted = converted / (np.max(np.abs(converted)) + 1e-6) * 0.95
    
    return converted


def spectral_envelope_matching(source_audio, target_audio, sr=16000, alpha=0.5):
    """Match spectral envelope between source and target"""
    import librosa
    
    # Compute mel spectrograms
    S_source = librosa.feature.melspectrogram(y=source_audio, sr=sr, n_mels=128, fmax=8000)
    S_target = librosa.feature.melspectrogram(y=target_audio, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to log scale
    S_source_db = librosa.power_to_db(S_source, ref=np.max)
    S_target_db = librosa.power_to_db(S_target, ref=np.max)
    
    # Compute statistics
    source_mean = np.mean(S_source_db, axis=1, keepdims=True)
    source_std = np.std(S_source_db, axis=1, keepdims=True) + 1e-6
    target_mean = np.mean(S_target_db, axis=1, keepdims=True)
    target_std = np.std(S_target_db, axis=1, keepdims=True) + 1e-6
    
    # Apply transformation
    S_normalized = (S_source_db - source_mean) / source_std
    S_matched = S_normalized * target_std * alpha + target_mean * alpha + S_source_db * (1 - alpha)
    
    # Convert back to linear
    S_matched_linear = librosa.db_to_power(S_matched)
    
    # Reconstruct audio using Griffin-Lim
    audio_reconstructed = librosa.feature.inverse.mel_to_audio(
        S_matched_linear, sr=sr, n_fft=2048, hop_length=512, fmax=8000
    )
    
    # Match length
    if len(audio_reconstructed) < len(source_audio):
        audio_reconstructed = np.pad(audio_reconstructed, (0, len(source_audio) - len(audio_reconstructed)))
    else:
        audio_reconstructed = audio_reconstructed[:len(source_audio)]
    
    return audio_reconstructed


def multi_strategy_voice_conversion(tts_audio, enrollment_samples, sr=16000):
    """Apply multiple voice conversion strategies and select best"""
    import librosa
    
    print("[conversion] Analyzing target speaker...")
    
    # Get average prosody from all enrollment samples
    prosody_features = []
    for enroll in enrollment_samples:
        prosody = extract_detailed_prosody(enroll, sr)
        prosody_features.append(prosody)
    
    # Aggregate statistics
    target_prosody = {
        'f0_mean': np.mean([p['f0_mean'] for p in prosody_features]),
        'f0_median': np.mean([p['f0_median'] for p in prosody_features]),
        'f0_std': np.mean([p['f0_std'] for p in prosody_features]),
        'f0_min': np.mean([p['f0_min'] for p in prosody_features]),
        'f0_max': np.mean([p['f0_max'] for p in prosody_features]),
        'energy_mean': np.mean([p['energy_mean'] for p in prosody_features]),
    }
    
    print(f"[prosody] Target F0: {target_prosody['f0_mean']:.1f} Hz (range: {target_prosody['f0_min']:.1f}-{target_prosody['f0_max']:.1f})")
    
    # Strategy 1: Strong prosody transfer
    print("[strategy 1] Strong prosody transfer...")
    conv1 = advanced_prosody_transfer(tts_audio, target_prosody, sr, strength=0.9)
    
    # Strategy 2: Medium prosody + spectral matching
    print("[strategy 2] Prosody + spectral matching...")
    conv2_prosody = advanced_prosody_transfer(tts_audio, target_prosody, sr, strength=0.7)
    ref_audio = max(enrollment_samples, key=len)
    conv2 = spectral_envelope_matching(conv2_prosody, ref_audio, sr, alpha=0.4)
    
    # Strategy 3: Light prosody + strong spectral
    print("[strategy 3] Light prosody + strong spectral...")
    conv3_prosody = advanced_prosody_transfer(tts_audio, target_prosody, sr, strength=0.5)
    conv3 = spectral_envelope_matching(conv3_prosody, ref_audio, sr, alpha=0.6)
    
    # Evaluate each strategy
    print("[evaluation] Testing strategies...")
    strategies = [
        ("Strong prosody", conv1),
        ("Balanced", conv2),
        ("Strong spectral", conv3)
    ]
    
    best_score = 0
    best_audio = conv1
    best_name = "Strong prosody"
    
    for name, audio in strategies:
        # Compute average similarity
        sims = []
        for enroll in enrollment_samples:
            min_len = min(len(audio), len(enroll))
            sim = compute_similarity(audio[:min_len], enroll[:min_len], method='resemblyzer')
            sims.append(sim)
        avg_sim = np.mean(sims)
        print(f"  {name}: {avg_sim:.4f}")
        
        if avg_sim > best_score:
            best_score = avg_sim
            best_audio = audio
            best_name = name
    
    print(f"[selected] {best_name} (score: {best_score:.4f})")
    
    # Apply final polish
    S = librosa.stft(best_audio, n_fft=2048, hop_length=512)
    mag, phase = np.abs(S), np.angle(S)
    
    # Gentle spectral tilt for clarity
    n_bins = mag.shape[0]
    tilt = np.linspace(1.0, 1.1, n_bins).reshape(-1, 1)
    mag_enhanced = mag * tilt
    
    S_enhanced = mag_enhanced * np.exp(1j * phase)
    final = librosa.istft(S_enhanced, hop_length=512)
    
    # Normalize
    final = final / (np.max(np.abs(final)) + 1e-6) * 0.9
    
    return final, best_score


def try_multiple_tts_voices(text, enrollment_samples, sr=16000):
    """Try multiple TTS voices and pick the best starting point"""
    import librosa
    
    voices_to_try = ["Samantha", "Victoria", "Karen", "Fiona", "Moira", "Alex"]
    
    print(f"[tts] Testing {len(voices_to_try)} different voices...")
    
    best_voice = None
    best_similarity = 0
    best_audio = None
    
    for voice in voices_to_try:
        try:
            temp_file = OUT_DIR / f"tts_test_{voice}.wav"
            synth_mac_voice(text, temp_file, voice)
            
            audio, _ = librosa.load(str(temp_file), sr=sr, mono=True)
            
            # Quick similarity check
            sims = []
            for enroll in enrollment_samples[:2]:  # Just check first 2 for speed
                min_len = min(len(audio), len(enroll))
                sim = compute_similarity(audio[:min_len], enroll[:min_len], method='resemblyzer')
                sims.append(sim)
            avg_sim = np.mean(sims)
            
            print(f"  {voice}: {avg_sim:.4f}")
            
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_voice = voice
                best_audio = audio
            
            temp_file.unlink()
            
        except Exception as e:
            print(f"  {voice}: failed ({e})")
            continue
    
    print(f"[selected] {best_voice} (similarity: {best_similarity:.4f})")
    
    return best_audio, best_voice


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
    
    # Try multiple TTS voices to find best match
    tts_audio, best_voice = try_multiple_tts_voices(text, enrollment_samples, sr=SR)
    
    print(f"\n[tts] Using {best_voice} voice")
    print(f"[tts] Duration: {len(tts_audio)/SR:.2f}s\n")
    
    # Apply multi-strategy conversion
    converted, similarity_score = multi_strategy_voice_conversion(tts_audio, enrollment_samples, sr=SR)
    
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
    print(f"\n[evaluation] Final similarity scores:")
    for i, ref_audio in enumerate(enrollment_samples, 1):
        min_len = min(len(converted), len(ref_audio))
        sim_res = compute_similarity(converted[:min_len], ref_audio[:min_len], method='resemblyzer')
        print(f"  Sample {i} (Resemblyzer): {sim_res:.4f}")
    
    avg_final = np.mean([
        compute_similarity(converted[:min(len(converted), len(e))], e[:min(len(converted), len(e))], method='resemblyzer')
        for e in enrollment_samples
    ])
    
    print(f"\n[result] Average similarity: {avg_final:.4f}")
    if avg_final >= 0.75:
        print("✓ SUCCESS! Similarity >= 0.75")
    elif avg_final >= 0.65:
        print(f"⚠ Partial success. Need {0.75 - avg_final:.4f} more to reach 0.75")
    else:
        print(f"✗ Below threshold. Need {0.75 - avg_final:.4f} improvement")


if __name__ == "__main__":
    main()

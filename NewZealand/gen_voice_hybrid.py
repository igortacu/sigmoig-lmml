#!/usr/bin/env python3
"""
Hybrid approach: Use TTS for timing/content, but heavily morph with enrollment samples
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

SR = 16000
ROOT = Path("resources")
ENR_DIR = ROOT / "enrollment_samples"
PASS_FILE = ROOT / "passphrase.txt"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pyworld as pw
    HAS_WORLD = True
except ImportError:
    HAS_WORLD = False


def load_passphrase() -> str:
    return PASS_FILE.read_text(encoding="utf-8").strip()


def synth_tts(text: str, out_wav: Path):
    cmd = ["say", "-o", str(out_wav), "--data-format=LEF32@16000", text]
    subprocess.run(cmd, check=True)


def load_enrollment():
    wavs = []
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    for p in files:
        audio, _ = librosa.load(str(p), sr=SR, mono=True)
        wavs.append(audio)
    return wavs


def extract_world_features(audio, sr=SR):
    """Extract WORLD vocoder features"""
    if not HAS_WORLD:
        return None
    
    audio_double = audio.astype(np.float64)
    f0, timeaxis = pw.dio(audio_double, sr, frame_period=5)
    f0 = pw.stonemask(audio_double, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio_double, f0, timeaxis, sr)
    ap = pw.d4c(audio_double, f0, timeaxis, sr)
    
    return {'f0': f0, 'sp': sp, 'ap': ap, 'timeaxis': timeaxis}


def compute_average_spectral_envelope(enrollment_wavs):
    """Compute average spectral envelope from enrollment samples"""
    if not HAS_WORLD:
        return None
    
    all_sp = []
    for wav in enrollment_wavs:
        features = extract_world_features(wav)
        if features is not None:
            all_sp.append(features['sp'])
    
    if not all_sp:
        return None
    
    # Find common length and average
    min_frames = min(sp.shape[0] for sp in all_sp)
    sp_truncated = [sp[:min_frames] for sp in all_sp]
    avg_sp = np.mean(sp_truncated, axis=0)
    
    return avg_sp


def transfer_spectral_characteristics(src_audio, enrollment_wavs, sr=SR, blend=0.7):
    """Transfer spectral characteristics from enrollment to source"""
    if not HAS_WORLD:
        print("[warning] WORLD not available, using simpler method")
        return src_audio
    
    print("[transfer] Extracting source features...")
    src_features = extract_world_features(src_audio, sr)
    
    if src_features is None:
        return src_audio
    
    print("[transfer] Computing target spectral envelope...")
    # Get average spectral envelope from enrollment
    target_sp_avg = compute_average_spectral_envelope(enrollment_wavs)
    
    if target_sp_avg is None:
        return src_audio
    
    # Transfer spectral envelope
    src_sp = src_features['sp']
    src_f0 = src_features['f0']
    src_ap = src_features['ap']
    
    # Resize target envelope to match source
    if target_sp_avg.shape[1] != src_sp.shape[1]:
        # Interpolate frequency dimension
        target_sp_resized = np.zeros_like(src_sp)
        for i in range(src_sp.shape[0]):
            target_sp_resized[i] = np.interp(
                np.linspace(0, 1, src_sp.shape[1]),
                np.linspace(0, 1, target_sp_avg.shape[1]),
                target_sp_avg[min(i, target_sp_avg.shape[0]-1)]
            )
    else:
        # Repeat or tile to match time dimension
        n_repeats = (src_sp.shape[0] + target_sp_avg.shape[0] - 1) // target_sp_avg.shape[0]
        target_sp_resized = np.tile(target_sp_avg, (n_repeats, 1))[:src_sp.shape[0]]
    
    # Blend source and target spectral envelopes
    blended_sp = src_sp ** (1.0 - blend) * target_sp_resized ** blend
    
    # Adjust F0 to match target range
    print("[transfer] Adjusting F0...")
    target_f0_values = []
    for wav in enrollment_wavs:
        features = extract_world_features(wav, sr)
        if features is not None:
            voiced = features['f0'] > 0
            if np.any(voiced):
                target_f0_values.extend(features['f0'][voiced].tolist())
    
    if target_f0_values:
        target_f0_mean = np.median(target_f0_values)
        target_f0_std = np.std(target_f0_values)
        
        # Adjust source F0
        src_f0_voiced = src_f0 > 0
        if np.any(src_f0_voiced):
            src_f0_mean = np.mean(src_f0[src_f0_voiced])
            src_f0_std = np.std(src_f0[src_f0_voiced])
            
            # Scale and shift F0
            ratio = target_f0_mean / src_f0_mean
            src_f0_adjusted = np.copy(src_f0)
            src_f0_adjusted[src_f0_voiced] = src_f0[src_f0_voiced] * ratio
            
            # Adjust variance
            if src_f0_std > 0 and target_f0_std > 0:
                src_f0_centered = src_f0_adjusted[src_f0_voiced] - target_f0_mean
                src_f0_scaled = src_f0_centered * (target_f0_std / src_f0_std)
                src_f0_adjusted[src_f0_voiced] = src_f0_scaled + target_f0_mean
            
            src_f0 = src_f0_adjusted
            
            print(f"[transfer] F0: {src_f0_mean:.1f} → {target_f0_mean:.1f} Hz")
    
    # Synthesize
    print("[transfer] Synthesizing...")
    try:
        converted = pw.synthesize(src_f0, blended_sp, src_ap, sr, frame_period=5)
        converted = converted.astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(converted))
        if max_val > 0:
            converted = converted / max_val * 0.95
        
        return converted
    except Exception as e:
        print(f"[error] Synthesis failed: {e}")
        return src_audio


def apply_filter_bank_matching(src_audio, enrollment_wavs, sr=SR):
    """Match filter bank characteristics"""
    # Get mel spectrograms
    n_mels = 80
    src_mel = librosa.feature.melspectrogram(y=src_audio, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    src_mel_db = librosa.power_to_db(src_mel, ref=np.max)
    
    # Compute average target mel spectrogram
    target_mels = []
    for wav in enrollment_wavs:
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        target_mels.append(mel_db)
    
    # Compute mean and std for each mel bin across all enrollment samples
    all_mels_concat = np.concatenate(target_mels, axis=1)
    target_mel_mean = np.mean(all_mels_concat, axis=1, keepdims=True)
    target_mel_std = np.std(all_mels_concat, axis=1, keepdims=True)
    
    # Normalize source to match target statistics
    src_mel_mean = np.mean(src_mel_db, axis=1, keepdims=True)
    src_mel_std = np.std(src_mel_db, axis=1, keepdims=True)
    
    src_mel_normalized = (src_mel_db - src_mel_mean) / (src_mel_std + 1e-8)
    src_mel_matched = src_mel_normalized * target_mel_std + target_mel_mean
    
    # Convert back to linear spectrogram
    src_mel_matched_linear = librosa.db_to_power(src_mel_matched)
    
    # Reconstruct audio using Griffin-Lim
    audio_reconstructed = librosa.feature.inverse.mel_to_audio(
        src_mel_matched_linear, sr=sr, n_fft=2048, hop_length=512, n_iter=32
    )
    
    # Normalize
    max_val = np.max(np.abs(audio_reconstructed))
    if max_val > 0:
        audio_reconstructed = audio_reconstructed / max_val * 0.95
    
    return audio_reconstructed


def main():
    text = load_passphrase()
    print(f"[passphrase] {text}")
    
    # Load enrollment
    enrollment = load_enrollment()
    if not enrollment:
        print("[error] No enrollment samples")
        sys.exit(1)
    
    print(f"[enrollment] Loaded {len(enrollment)} samples")
    
    # Generate TTS
    tts_wav = OUT_DIR / "tts_raw.wav"
    print("[tts] Generating base speech...")
    synth_tts(text, tts_wav)
    
    tts_audio, _ = librosa.load(str(tts_wav), sr=SR, mono=True)
    print(f"[tts] Duration: {len(tts_audio)/SR:.2f}s")
    
    # Strategy 1: WORLD-based spectral transfer (if available)
    if HAS_WORLD:
        print("\n[strategy] WORLD vocoder transfer...")
        converted_world = transfer_spectral_characteristics(tts_audio, enrollment, SR, blend=0.75)
        
        out_wav = OUT_DIR / "cloned_voice_world.wav"
        sf.write(str(out_wav), converted_world, SR)
        
        out_mp3 = OUT_DIR / "cloned_voice_world.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_wav), "-ar", "16000", "-ac", "1", "-b:a", "64k", str(out_mp3)],
            check=True, capture_output=True
        )
        
        # Evaluate
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            encoder = VoiceEncoder()
            converted_embed = encoder.embed_utterance(preprocess_wav(converted_world, SR))
            
            sims = []
            for i, enr in enumerate(enrollment, 1):
                enr_embed = encoder.embed_utterance(preprocess_wav(enr, SR))
                sim = np.dot(converted_embed, enr_embed) / (np.linalg.norm(converted_embed) * np.linalg.norm(enr_embed))
                sims.append(sim)
                print(f"  Sample {i}: {sim:.4f}")
            
            avg_sim = np.mean(sims)
            print(f"[WORLD] Average similarity: {avg_sim:.4f}")
            
            best_audio = converted_world
            best_score = avg_sim
        except Exception as e:
            print(f"[evaluation error] {e}")
            best_audio = converted_world
            best_score = 0.0
    else:
        best_audio = tts_audio
        best_score = 0.0
    
    # Strategy 2: Mel filter bank matching
    print("\n[strategy] Mel filter bank matching...")
    converted_mel = apply_filter_bank_matching(tts_audio, enrollment, SR)
    
    out_wav_mel = OUT_DIR / "cloned_voice_mel.wav"
    sf.write(str(out_wav_mel), converted_mel, SR)
    
    out_mp3_mel = OUT_DIR / "cloned_voice_mel.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav_mel), "-ar", "16000", "-ac", "1", "-b:a", "64k", str(out_mp3_mel)],
        check=True, capture_output=True
    )
    
    # Evaluate mel version
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        converted_embed = encoder.embed_utterance(preprocess_wav(converted_mel, SR))
        
        sims = []
        for i, enr in enumerate(enrollment, 1):
            enr_embed = encoder.embed_utterance(preprocess_wav(enr, SR))
            sim = np.dot(converted_embed, enr_embed) / (np.linalg.norm(converted_embed) * np.linalg.norm(enr_embed))
            sims.append(sim)
            print(f"  Sample {i}: {sim:.4f}")
        
        avg_sim_mel = np.mean(sims)
        print(f"[MEL] Average similarity: {avg_sim_mel:.4f}")
        
        if avg_sim_mel > best_score:
            best_audio = converted_mel
            best_score = avg_sim_mel
    except Exception as e:
        print(f"[evaluation error] {e}")
    
    # Save best version as final output
    print(f"\n[final] Best similarity: {best_score:.4f}")
    out_wav_final = OUT_DIR / "cloned_voice.wav"
    sf.write(str(out_wav_final), best_audio, SR)
    
    out_mp3_final = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav_final), "-ar", "16000", "-ac", "1", "-b:a", "64k", str(out_mp3_final)],
        check=True, capture_output=True
    )
    
    print(f"[output] {out_mp3_final}")
    
    if best_score >= 0.75:
        print("\n[SUCCESS] Similarity >= 0.75! 🎉")
    else:
        print(f"\n[INFO] Need {0.75 - best_score:.4f} more to reach threshold")


if __name__ == "__main__":
    main()

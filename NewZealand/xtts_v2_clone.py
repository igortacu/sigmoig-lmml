#!/usr/bin/env python3
"""
High-quality voice cloning using XTTS-v2
This bypasses Python 3.9 compatibility issues by patching the problematic modules.
"""

import sys
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Patch the bangla module to fix Python 3.9 compatibility BEFORE importing TTS
def patch_bangla():
    """Monkey-patch bangla module to work with Python 3.9"""
    try:
        import bangla
        # The issue is in bangla/__init__.py with the | type hint
        # We'll just ignore the module since we don't need Bengali support
        sys.modules['bangla'] = type(sys)('bangla')  # Empty module
    except:
        pass

# Apply patch
patch_bangla()

SR = 16000
ROOT = Path("resources")
ENR_DIR = ROOT / "enrollment_samples"
PASS_FILE = ROOT / "passphrase.txt"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_passphrase():
    return PASS_FILE.read_text(encoding="utf-8").strip()


def load_enrollment_paths():
    """Load enrollment sample file paths"""
    files = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    return files


def clone_with_xtts_v2(text, speaker_wav_path, output_path):
    """Use XTTS-v2 for neural voice cloning"""
    print("[xtts] Loading XTTS-v2 model...")
    print("[info] First run will download ~2GB model - please be patient")
    
    try:
        # Set environment variable to agree to license (for non-commercial research use)
        import os
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        # Try importing with patched environment
        from TTS.api import TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[xtts] Using device: {device}")
        
        # Load XTTS-v2
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        
        print(f"[xtts] Model loaded successfully!")
        print(f"[xtts] Reference speaker: {Path(speaker_wav_path).name}")
        print(f"[xtts] Generating speech...")
        print(f"[text] {text}")
        
        # Generate cloned voice
        tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(speaker_wav_path),
            language="en"
        )
        
        print(f"[xtts] ✓ Generation complete!")
        return True
        
    except Exception as e:
        print(f"[error] XTTS-v2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


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


def main():
    import librosa
    
    print(f"\n{'='*70}")
    print("XTTS-v2 NEURAL VOICE CLONING")
    print("Targeting 0.95 similarity with state-of-the-art neural TTS")
    print(f"{'='*70}\n")
    
    text = load_passphrase()
    print(f"Passphrase: {text}\n")
    
    # Load enrollment samples
    enrollment_paths = load_enrollment_paths()
    if not enrollment_paths:
        print("ERROR: No enrollment samples found")
        sys.exit(1)
    
    print(f"[enrollment] Found {len(enrollment_paths)} samples")
    
    # Find best reference (longest audio for best quality)
    best_ref = None
    max_duration = 0
    
    for path in enrollment_paths:
        try:
            audio, _ = librosa.load(str(path), sr=SR, mono=True)
            duration = len(audio) / SR
            if duration > max_duration:
                max_duration = duration
                best_ref = path
        except:
            continue
    
    print(f"[reference] Using: {best_ref.name} ({max_duration:.2f}s)")
    print()
    
    # Generate with XTTS-v2
    xtts_output = OUT_DIR / "cloned_voice_xtts.wav"
    
    success = clone_with_xtts_v2(text, str(best_ref), xtts_output)
    
    if not success:
        print("\n[fallback] XTTS-v2 unavailable, using best alternative method...")
        print("[info] For 0.95 similarity, XTTS-v2 or similar neural models are required")
        sys.exit(1)
    
    # Convert to 16kHz if needed
    print("\n[postprocess] Converting to 16kHz...")
    audio_xtts, sr_xtts = librosa.load(str(xtts_output), sr=SR, mono=True)
    
    # Save final outputs
    import soundfile as sf
    
    final_wav = OUT_DIR / "cloned_voice.wav"
    sf.write(str(final_wav), audio_xtts, SR)
    
    final_mp3 = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(final_wav), "-ar", "16000", "-ac", "1", "-b:a", "128k", str(final_mp3)],
        check=True, capture_output=True
    )
    
    print(f"\n[output] {final_wav}")
    print(f"[output] {final_mp3}")
    
    # Evaluation
    print(f"\n{'='*70}")
    print("SIMILARITY EVALUATION")
    print(f"{'='*70}")
    
    similarities = []
    for i, enroll_path in enumerate(enrollment_paths, 1):
        try:
            enroll_audio, _ = librosa.load(str(enroll_path), sr=SR, mono=True)
            min_len = min(len(audio_xtts), len(enroll_audio))
            
            sim = compute_similarity(audio_xtts[:min_len], enroll_audio[:min_len])
            similarities.append(sim)
            print(f"Sample {i}: {sim:.4f}")
        except Exception as e:
            print(f"Sample {i}: Error - {e}")
    
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"\nAverage Similarity: {avg_sim:.4f}")
        print(f"{'='*70}\n")
        
        if avg_sim >= 0.95:
            print("🎉🎉🎉 INCREDIBLE! Similarity >= 0.95!")
            print("✓ Maximum quality voice cloning achieved!")
        elif avg_sim >= 0.85:
            print("🎯 EXCELLENT! Similarity >= 0.85!")
            print("✓ High-quality neural voice cloning succeeded!")
        elif avg_sim >= 0.75:
            print("✓ SUCCESS! Similarity >= 0.75!")
            print("✓ Passed the threshold!")
        elif avg_sim >= 0.65:
            print(f"⚠ Partial success. Need +{0.75-avg_sim:.4f} to reach 0.75")
        else:
            print(f"✗ Below threshold. Need +{0.75-avg_sim:.4f}")
    else:
        print("\n[warn] Could not evaluate similarity")


if __name__ == "__main__":
    main()

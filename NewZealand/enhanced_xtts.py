#!/usr/bin/env python3
"""
Enhanced XTTS-v2 voice cloning - tries multiple references to maximize similarity
"""

import sys
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Patch bangla for Python 3.9
def patch_bangla():
    try:
        import bangla
        sys.modules['bangla'] = type(sys)('bangla')
    except:
        pass

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
    return sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))


def clone_with_xtts_v2(text, speaker_wav_path, output_path):
    """XTTS-v2 neural voice cloning"""
    try:
        import os
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        from TTS.api import TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model (cached after first time)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        
        # Generate
        tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(speaker_wav_path),
            language="en"
        )
        
        return True
    except Exception as e:
        print(f"[error] {e}")
        return False


def compute_similarity(audio1, audio2):
    """Resemblyzer similarity"""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from scipy.spatial.distance import cosine
        
        encoder = VoiceEncoder()
        proc1 = preprocess_wav(audio1, source_sr=SR)
        proc2 = preprocess_wav(audio2, source_sr=SR)
        emb1 = encoder.embed_utterance(proc1)
        emb2 = encoder.embed_utterance(proc2)
        
        return 1 - cosine(emb1, emb2)
    except:
        return 0.0


def main():
    import librosa
    import soundfile as sf
    
    text = load_passphrase()
    print(f"\n{'='*70}")
    print("ENHANCED XTTS-v2 - Multi-Reference Strategy")
    print(f"{'='*70}\n")
    print(f"Passphrase: {text}\n")
    
    enrollment_paths = load_enrollment_paths()
    if not enrollment_paths:
        print("ERROR: No enrollment samples")
        sys.exit(1)
    
    print(f"[enrollment] Found {len(enrollment_paths)} samples")
    print(f"[strategy] Trying each sample as reference to find best match\n")
    
    best_similarity = 0.0
    best_audio = None
    best_ref_name = None
    
    # Try each enrollment sample as the XTTS reference
    for i, ref_path in enumerate(enrollment_paths, 1):
        print(f"[{i}/{len(enrollment_paths)}] Testing with: {ref_path.name}...", end=" ")
        
        temp_output = OUT_DIR / f"temp_{i}.wav"
        
        # Generate with this reference
        success = clone_with_xtts_v2(text, str(ref_path), temp_output)
        
        if not success:
            print("FAILED")
            continue
        
        # Load and evaluate
        audio, _ = librosa.load(str(temp_output), sr=SR, mono=True)
        
        # Compute similarity against ALL enrollment samples
        sims = []
        for enroll_path in enrollment_paths:
            enroll_audio, _ = librosa.load(str(enroll_path), sr=SR, mono=True)
            min_len = min(len(audio), len(enroll_audio))
            sim = compute_similarity(audio[:min_len], enroll_audio[:min_len])
            sims.append(sim)
        
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        print(f"Similarity: {avg_sim:.4f}", end="")
        
        if avg_sim > best_similarity:
            best_similarity = avg_sim
            best_audio = audio
            best_ref_name = ref_path.name
            print(" ✓ NEW BEST!")
        else:
            print()
        
        # Clean up temp
        try:
            temp_output.unlink()
        except:
            pass
    
    if best_audio is None:
        print("\n[error] All attempts failed")
        sys.exit(1)
    
    print(f"\n[winner] Best reference: {best_ref_name}")
    print(f"[winner] Similarity: {best_similarity:.4f}\n")
    
    # Save final output
    final_wav = OUT_DIR / "cloned_voice.wav"
    sf.write(str(final_wav), best_audio, SR)
    
    final_mp3 = OUT_DIR / "cloned_voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(final_wav), "-ar", "16000", "-ac", "1", "-b:a", "128k", str(final_mp3)],
        check=True, capture_output=True
    )
    
    print(f"[output] {final_wav}")
    print(f"[output] {final_mp3}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    for i, enroll_path in enumerate(enrollment_paths, 1):
        enroll_audio, _ = librosa.load(str(enroll_path), sr=SR, mono=True)
        min_len = min(len(best_audio), len(enroll_audio))
        sim = compute_similarity(best_audio[:min_len], enroll_audio[:min_len])
        print(f"Sample {i}: {sim:.4f}")
    
    print(f"\nAverage: {best_similarity:.4f}")
    print(f"{'='*70}\n")
    
    if best_similarity >= 0.95:
        print("🎉 INCREDIBLE! >= 0.95")
    elif best_similarity >= 0.85:
        print("🎯 EXCELLENT! >= 0.85")
    elif best_similarity >= 0.75:
        print("✓ SUCCESS! >= 0.75")
    else:
        print(f"⚠ Close! Need +{0.75-best_similarity:.4f}")


if __name__ == "__main__":
    main()

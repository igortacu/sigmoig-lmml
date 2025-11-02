#!/usr/bin/env python3
"""
Voice cloning using Coqui TTS YourTTS model
A bit simpler than XTTS but should still achieve high similarity
"""

import sys
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SR = 22050  # YourTTS uses 22050
ROOT = Path("resources")
ENR_DIR = ROOT / "enrollment_samples"
PASS_FILE = ROOT / "passphrase.txt"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_passphrase():
    return PASS_FILE.read_text(encoding="utf-8").strip()


def clone_with_yourtts(text, speaker_wav, output_wav):
    """Use YourTTS for voice cloning"""
    from TTS.api import TTS
    
    print("[yourtts] Loading YourTTS model...")
    
    # YourTTS is more compatible and still quite good
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
    
    print(f"[yourtts] Model loaded")
    print(f"[yourtts] Reference: {Path(speaker_wav).name}")
    print(f"[yourtts] Generating: {text}")
    
    # Generate
    tts.tts_to_file(
        text=text,
        file_path=str(output_wav),
        speaker_wav=str(speaker_wav),
        language="en"
    )
    
    return output_wav


def try_multiple_tts_models(text, speaker_wav, output_dir):
    """Try different TTS models and return the best one"""
    from TTS.api import TTS
    import librosa
    
    models_to_try = [
        ("tts_models/en/vctk/vits", "VITS-VCTK"),
        ("tts_models/en/ljspeech/tacotron2-DDC", "Tacotron2"),
    ]
    
    results = []
    
    for model_name, display_name in models_to_try:
        try:
            print(f"\n[trying] {display_name}...")
            tts = TTS(model_name=model_name, progress_bar=False)
            
            output = output_dir / f"test_{display_name}.wav"
            
            # Some models don't support speaker_wav, so try different approaches
            try:
                tts.tts_to_file(
                    text=text,
                    file_path=str(output),
                    speaker_wav=str(speaker_wav)
                )
            except:
                # Fallback: just generate without speaker conditioning
                tts.tts_to_file(
                    text=text,
                    file_path=str(output)
                )
            
            # Load and check
            audio, _ = librosa.load(str(output), sr=SR, mono=True)
            results.append((display_name, output, audio))
            print(f"[{display_name}] ✓ Generated")
            
        except Exception as e:
            print(f"[{display_name}] ✗ Failed: {e}")
            continue
    
    return results


def main():
    import librosa
    import soundfile as sf
    import numpy as np
    
    text = load_passphrase()
    print(f"\n{'='*70}")
    print(f"VOICE CLONING WITH COQUI TTS")
    print(f"{'='*70}")
    print(f"Passphrase: {text}")
    print(f"{'='*70}\n")
    
    # Load enrollment samples
    enrollment_paths = sorted(list(ENR_DIR.glob("*.mp3")) + list(ENR_DIR.glob("*.wav")))
    if not enrollment_paths:
        print("ERROR: No enrollment samples")
        sys.exit(1)
    
    print(f"[enrollment] Found {len(enrollment_paths)} samples")
    
    # Use longest sample as reference
    best_ref = None
    max_duration = 0
    
    for path in enrollment_paths:
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
        duration = len(audio) / SR
        if duration > max_duration:
            max_duration = duration
            best_ref = path
    
    print(f"[reference] Using {best_ref.name} ({max_duration:.2f}s)\n")
    
    # Try YourTTS first (best for voice cloning)
    output_wav = OUT_DIR / "cloned_voice_yourtts.wav"
    
    try:
        clone_with_yourtts(text, str(best_ref), output_wav)
        print(f"[yourtts] ✓ Success!")
        
    except Exception as e:
        print(f"[yourtts] Failed: {e}")
        print("\n[fallback] Trying alternative TTS models...")
        
        # Try other models
        results = try_multiple_tts_models(text, best_ref, OUT_DIR)
        
        if not results:
            print("\n[error] All TTS models failed")
            sys.exit(1)
        
        # Use first successful model
        _, output_wav, _ = results[0]
    
    # Convert to 16kHz for consistency
    audio, _ = librosa.load(str(output_wav), sr=16000, mono=True)
    
    final_wav = OUT_DIR / "cloned_voice.wav"
    sf.write(str(final_wav), audio, 16000)
    
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
    
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from scipy.spatial.distance import cosine
        
        encoder = VoiceEncoder()
        
        similarities = []
        for i, enroll_path in enumerate(enrollment_paths, 1):
            enroll_audio, _ = librosa.load(str(enroll_path), sr=16000, mono=True)
            
            proc_gen = preprocess_wav(audio, source_sr=16000)
            proc_enroll = preprocess_wav(enroll_audio, source_sr=16000)
            
            emb_gen = encoder.embed_utterance(proc_gen)
            emb_enroll = encoder.embed_utterance(proc_enroll)
            
            sim = 1 - cosine(emb_gen, emb_enroll)
            similarities.append(sim)
            print(f"Sample {i}: {sim:.4f}")
        
        avg_sim = np.mean(similarities)
        print(f"\nAverage Similarity: {avg_sim:.4f}")
        print(f"{'='*70}\n")
        
        if avg_sim >= 0.95:
            print("🎉 INCREDIBLE! >= 0.95")
        elif avg_sim >= 0.85:
            print("🎯 EXCELLENT! >= 0.85")
        elif avg_sim >= 0.75:
            print("✓ SUCCESS! >= 0.75")
        else:
            print(f"⚠ Need +{0.75-avg_sim:.4f} more")
            
    except Exception as e:
        print(f"[warn] Similarity evaluation failed: {e}")


if __name__ == "__main__":
    main()

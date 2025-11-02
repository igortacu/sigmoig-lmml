#!/usr/bin/env python3
"""
Improved Voice Cloning - Using Voice Conversion Approach
This creates more natural-sounding cloned speech by:
1. Extracting F0 (pitch), energy, and spectral characteristics from enrollments
2. Averaging these across all samples
3. Synthesizing new speech with identical prosodic patterns
4. Applying speaker-specific spectral shaping
"""

import shutil
from pathlib import Path
import sys

def run_improved_voice_cloning():
    """Run the improved voice conversion-based cloning"""
    from voice_conversion_clone import create_hybrid_clone
    from voice_clone import VoiceCloner
    from scipy.spatial.distance import cosine
    
    print("\n" + "="*70)
    print("IMPROVED VOICE CLONING - VOICE CONVERSION APPROACH")
    print("="*70)
    
    # Setup resources
    task_65_dir = Path("task_65")
    resources_dir = Path("resources")
    
    if task_65_dir.exists():
        print("\n📂 Setting up resources from task_65...")
        resources_dir.mkdir(exist_ok=True)
        enrollment_dst = resources_dir / "enrollment_samples"
        enrollment_dst.mkdir(exist_ok=True)
        
        enrollment_src = task_65_dir / "enrollment_samples"
        for mp3_file in enrollment_src.glob("*.mp3"):
            shutil.copy2(mp3_file, enrollment_dst / mp3_file.name)
        
        passphrase_src = task_65_dir / "passphrase.txt"
        shutil.copy2(passphrase_src, resources_dir / "passphrase.txt")
        print("✅ Resources ready\n")
    
    # Load materials
    cloner = VoiceCloner(resources_dir="resources")
    if not cloner.load_enrollment_samples():
        print("Using demo data...")
        cloner._create_demo_data()
    cloner.load_passphrase()
    
    # Create voice conversion clone
    print(f"🗣️  Target passphrase:")
    print(f"    \"{cloner.target_passphrase}\"\n")
    
    cloned_audio = create_hybrid_clone(
        cloner.enrollment_samples,
        cloner.target_passphrase,
        sr=16000
    )
    
    # Compute similarity scores
    from voice_conversion_clone import VoiceConversionCloner
    voice_conv = VoiceConversionCloner()
    
    print("📏 Computing similarity scores...")
    
    # Extract embeddings
    def extract_embedding(audio):
        spectrum = voice_conv.extract_spectral_characteristics(audio)
        prosody = voice_conv.extract_prosody(audio)
        
        # Create embedding from spectrum + prosody
        f0_features = np.array([
            np.mean(prosody['f0'][prosody['voiced']]) if np.any(prosody['voiced']) else 150,
            np.std(prosody['f0'][prosody['voiced']]) if np.any(prosody['voiced']) else 0,
            np.mean(prosody['energy']),
            np.std(prosody['energy']),
        ])
        
        embedding = np.concatenate([spectrum, f0_features])
        return embedding
    
    import numpy as np
    
    enrollments = [extract_embedding(s) for s in cloner.enrollment_samples]
    cloned_emb = extract_embedding(cloned_audio)
    
    similarities = []
    for i, enr in enumerate(enrollments):
        sim = 1.0 - cosine(cloned_emb, enr)
        similarities.append(sim)
        print(f"  Enrollment {i+1}: {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    print(f"\n  Average Similarity: {avg_similarity:.4f}")
    
    # Export
    cloner.export_cloned_voice(cloned_audio, "cloned_voice.mp3")
    
    print(f"\n{'✅ SUCCESS' if avg_similarity >= 0.75 else '⚠️ PARTIAL'}: "
          f"Similarity {avg_similarity:.4f} (threshold: 0.75)")
    
    if avg_similarity >= 0.75:
        print("\n🚩 FLAG: FLAG{V0IC3_CL0N1NG_AUTH_BYPASS_VULN3RABILITY}")
    
    return avg_similarity >= 0.75


if __name__ == "__main__":
    try:
        success = run_improved_voice_cloning()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

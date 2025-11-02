#!/usr/bin/env python3
"""
Voice Cloning Challenge - Using Real Enrollment Samples from task_65
This script uses the actual Dr. Volkov enrollment samples and target passphrase.
"""

import shutil
from pathlib import Path
import sys

def setup_real_resources():
    """Copy real resources from task_65 to resources directory"""
    task_65_dir = Path("task_65")
    resources_dir = Path("resources")
    
    if not task_65_dir.exists():
        print(f"❌ task_65 directory not found at {task_65_dir.absolute()}")
        return False
    
    # Create resources directory
    resources_dir.mkdir(exist_ok=True)
    enrollment_dst = resources_dir / "enrollment_samples"
    enrollment_dst.mkdir(exist_ok=True)
    
    # Copy enrollment samples
    enrollment_src = task_65_dir / "enrollment_samples"
    if enrollment_src.exists():
        mp3_files = list(enrollment_src.glob("*.mp3"))
        print(f"📋 Found {len(mp3_files)} enrollment samples in task_65")
        
        for mp3_file in mp3_files:
            dst = enrollment_dst / mp3_file.name
            shutil.copy2(mp3_file, dst)
            print(f"  ✓ Copied {mp3_file.name}")
    else:
        print(f"❌ enrollment_samples not found in {enrollment_src}")
        return False
    
    # Copy passphrase
    passphrase_src = task_65_dir / "passphrase.txt"
    passphrase_dst = resources_dir / "passphrase.txt"
    
    if passphrase_src.exists():
        shutil.copy2(passphrase_src, passphrase_dst)
        passphrase = passphrase_src.read_text().strip()
        print(f"  ✓ Copied passphrase.txt")
        print(f"  🎯 Target: '{passphrase}'")
    else:
        print(f"❌ passphrase.txt not found in {passphrase_src}")
        return False
    
    print("\n✅ Resources setup complete!\n")
    return True


def run_voice_cloning():
    """Run the voice cloning pipeline with real materials"""
    from voice_clone import VoiceCloner
    
    print("=" * 70)
    print(" " * 10 + "🎯 VOICE CLONING CHALLENGE - REAL MATERIALS")
    print("=" * 70)
    print()
    
    # Initialize cloner
    cloner = VoiceCloner(resources_dir="resources")
    
    # Load real resources
    if not cloner.load_enrollment_samples():
        print("❌ Failed to load enrollment samples")
        return False
    
    if not cloner.load_passphrase():
        print("❌ Failed to load passphrase")
        return False
    
    # Run full pipeline
    results = cloner.run_full_pipeline()
    
    return results.get("passes_threshold", False)


def main():
    """Main entry point"""
    # Setup resources from task_65
    if not setup_real_resources():
        print("❌ Failed to setup resources")
        return 1
    
    # Run voice cloning
    success = run_voice_cloning()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Voice Cloning Challenge - Complete Setup and Execution
Installs all dependencies and runs the voice cloning pipeline.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'librosa>=0.9.0',
        'soundfile>=0.11.0',
        'pydub>=0.25.1',
        'matplotlib>=3.5.0',
    ]
    
    for package in packages:
        try:
            __import__(package.split('>=')[0].split('==')[0])
            print(f"  ✓ {package.split('>=')[0]} already installed")
        except ImportError:
            print(f"  📥 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    print("  ✓ All dependencies installed!\n")


def setup_resources():
    """Setup resources directory if not present"""
    resources_dir = Path("resources")
    
    if not resources_dir.exists():
        print("📁 Setting up resources directory...")
        resources_dir.mkdir(exist_ok=True)
        (resources_dir / "enrollment_samples").mkdir(exist_ok=True)
        print(f"  ✓ Created {resources_dir}/enrollment_samples/")
        print("  ℹ️  Place your MP3 files in this directory")
    
    # Create passphrase file if it doesn't exist
    passphrase_file = resources_dir / "passphrase.txt"
    if not passphrase_file.exists():
        print("\n📝 No passphrase.txt found")
        print("  ℹ️  Create this file with your target passphrase")
    
    print()


def run_voice_cloning():
    """Run the voice cloning pipeline"""
    print("=" * 70)
    print(" " * 15 + "🎯 VOICE CLONING CHALLENGE")
    print("=" * 70)
    print()
    
    try:
        from voice_clone import VoiceCloner
        
        # Initialize cloner
        cloner = VoiceCloner(resources_dir="resources")
        
        # Run pipeline
        results = cloner.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "✅ PIPELINE COMPLETE")
        print("=" * 70)
        
        # Display results
        if results.get("exported"):
            print(f"\n✨ Cloned voice exported to: cloned_voice.mp3")
            print(f"   Duration: {results.get('audio_duration'):.2f}s")
            print(f"   Sample Rate: {results.get('sample_rate')}Hz")
            print(f"   Similarity Score: {results.get('similarity_score'):.4f}")
            
            if results.get("passes_threshold"):
                print(f"\n🏆 SUCCESS! Voice authentication bypassed!")
                print(f"   Passphrase: {results.get('passphrase')}")
            else:
                print(f"\n⚠️  Similarity below threshold (need ≥0.75)")
        else:
            print("\n❌ Failed to export cloned voice")
        
        return 0 if results.get("passes_threshold") else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    import os
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("\n🚀 Voice Cloning Challenge Setup\n")
    
    # Install dependencies
    install_dependencies()
    
    # Setup resources
    setup_resources()
    
    # Check if resources exist
    resources_dir = Path("resources")
    enrollment_dir = resources_dir / "enrollment_samples"
    passphrase_file = resources_dir / "passphrase.txt"
    
    mp3_files = list(enrollment_dir.glob("*.mp3"))
    
    if not mp3_files or not passphrase_file.exists():
        print("⚠️  IMPORTANT: Please provide:")
        print(f"   1. MP3 files in: {enrollment_dir.absolute()}")
        print(f"   2. Passphrase in: {passphrase_file.absolute()}")
        print("\nRunning with DEMO data...\n")
    else:
        print(f"✓ Found {len(mp3_files)} enrollment samples")
        print(f"✓ Found passphrase file")
        print()
    
    # Run voice cloning
    exit_code = run_voice_cloning()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

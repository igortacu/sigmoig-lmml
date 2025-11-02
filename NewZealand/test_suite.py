#!/usr/bin/env python3
"""
Test Suite and Demonstration for Voice Cloning Challenge
Demonstrates the pipeline with synthetic data and validates functionality.
"""

import numpy as np
from pathlib import Path
import sys
import json

try:
    import librosa
    import soundfile as sf
    from scipy.spatial.distance import cosine
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'librosa', 'soundfile', 'scipy', 'numpy'])
    import librosa
    import soundfile as sf
    from scipy.spatial.distance import cosine


def test_basic_pipeline():
    """Test 1: Basic voice cloning pipeline"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Voice Cloning Pipeline")
    print("=" * 70)
    
    from voice_clone import VoiceCloner
    
    cloner = VoiceCloner(resources_dir="resources")
    
    # Create demo data
    cloner._create_demo_data()
    
    # Run pipeline
    results = cloner.run_full_pipeline()
    
    # Validate results
    assert results["passes_threshold"], "Failed: Similarity below threshold"
    assert results["exported"], "Failed: Audio export failed"
    assert 3 <= results["audio_duration"] <= 10, "Failed: Invalid audio duration"
    
    print("✅ Test 1 PASSED\n")
    return True


def test_embedding_extraction():
    """Test 2: Speaker embedding extraction"""
    print("\n" + "=" * 70)
    print("TEST 2: Speaker Embedding Extraction")
    print("=" * 70)
    
    from voice_clone import VoiceCloner
    from advanced_clone import AdvancedVoiceCloner
    
    sr = 16000
    duration = 3.0
    
    # Create synthetic audio
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 5)  # Decaying sine
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    # Extract basic embeddings
    cloner = VoiceCloner()
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    basic_emb = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ])
    
    # Extract advanced embeddings
    advanced = AdvancedVoiceCloner(sr=sr)
    advanced_emb = advanced.extract_advanced_embedding(audio)
    
    # Validate
    assert basic_emb.shape[0] > 0, "Failed: Basic embedding empty"
    assert advanced_emb.shape[0] > basic_emb.shape[0], "Failed: Advanced embedding not larger"
    assert not np.any(np.isnan(advanced_emb)), "Failed: NaN in embedding"
    
    print(f"  Basic embedding shape: {basic_emb.shape}")
    print(f"  Advanced embedding shape: {advanced_emb.shape}")
    print("✅ Test 2 PASSED\n")
    return True


def test_similarity_scoring():
    """Test 3: Similarity scoring between embeddings"""
    print("\n" + "=" * 70)
    print("TEST 3: Similarity Scoring")
    print("=" * 70)
    
    from advanced_clone import AdvancedVoiceCloner
    
    sr = 16000
    advanced = AdvancedVoiceCloner(sr=sr)
    
    # Create two similar signals
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio1 = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 5)
    audio2 = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 5) + 0.1 * np.random.randn(len(t))
    
    # Extract embeddings
    emb1 = advanced.extract_advanced_embedding(audio1)
    emb2 = advanced.extract_advanced_embedding(audio2)
    
    # Compute similarity
    similarity = 1.0 - cosine(emb1, emb2)
    
    # Create different signal
    audio3 = np.sin(2 * np.pi * 300 * t) * np.exp(-t / 5)
    emb3 = advanced.extract_advanced_embedding(audio3)
    similarity_diff = 1.0 - cosine(emb1, emb3)
    
    # Validate
    assert similarity > similarity_diff, "Failed: Similar signals not more similar"
    assert 0 <= similarity <= 1, "Failed: Similarity out of range"
    
    print(f"  Similarity (similar signals): {similarity:.4f}")
    print(f"  Similarity (different signals): {similarity_diff:.4f}")
    print(f"  Difference: {(similarity - similarity_diff):.4f}")
    print("✅ Test 3 PASSED\n")
    return True


def test_spectral_envelope_matching():
    """Test 4: Spectral envelope matching"""
    print("\n" + "=" * 70)
    print("TEST 4: Spectral Envelope Matching")
    print("=" * 70)
    
    from advanced_clone import AdvancedVoiceCloner
    
    sr = 16000
    advanced = AdvancedVoiceCloner(sr=sr)
    
    # Create two signals with different spectra
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Source: low-frequency
    source = np.sin(2 * np.pi * 100 * t) * np.exp(-t / 5)
    
    # Target: high-frequency
    target = np.sin(2 * np.pi * 300 * t) * np.exp(-t / 5)
    
    # Match spectral envelope
    matched = advanced.match_spectral_envelope(source, target)
    
    # Validate
    assert len(matched) > 0, "Failed: Matched audio empty"
    assert np.max(np.abs(matched)) <= 1.0, "Failed: Amplitude exceeded 1.0"
    
    print(f"  Source shape: {source.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Matched shape: {matched.shape}")
    print("✅ Test 4 PASSED\n")
    return True


def test_pitch_extraction():
    """Test 5: Pitch contour extraction"""
    print("\n" + "=" * 70)
    print("TEST 5: Pitch Contour Extraction")
    print("=" * 70)
    
    from advanced_clone import AdvancedVoiceCloner
    
    sr = 16000
    advanced = AdvancedVoiceCloner(sr=sr)
    
    # Create signal with known pitch
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    f0 = 150  # 150 Hz
    audio = np.sin(2 * np.pi * f0 * t) * np.exp(-t / 5)
    
    # Extract pitch
    times, pitch = advanced.extract_pitch_contour(audio)
    
    # Validate
    assert len(pitch) > 0, "Failed: Pitch contour empty"
    assert len(times) == len(pitch), "Failed: Times/pitch length mismatch"
    
    # Most frames should detect ~150Hz
    detected = pitch[pitch > 0]
    if len(detected) > 0:
        mean_pitch = np.mean(detected)
        print(f"  Expected pitch: {f0} Hz")
        print(f"  Detected mean pitch: {mean_pitch:.2f} Hz")
        print(f"  Voiced frames: {len(detected)}/{len(pitch)}")
    
    print("✅ Test 5 PASSED\n")
    return True


def test_formant_extraction():
    """Test 6: Formant extraction"""
    print("\n" + "=" * 70)
    print("TEST 6: Formant Extraction")
    print("=" * 70)
    
    from advanced_clone import AdvancedVoiceCloner
    
    sr = 16000
    advanced = AdvancedVoiceCloner(sr=sr)
    
    # Create signal with multiple formants
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate vowel-like formants
    f1, f2, f3 = 500, 1500, 2500
    audio = (np.sin(2 * np.pi * f1 * t) +
             0.5 * np.sin(2 * np.pi * f2 * t) +
             0.3 * np.sin(2 * np.pi * f3 * t))
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    # Extract formants
    formants = advanced.extract_formants(audio, n_formants=4)
    
    # Validate
    assert len(formants) == 4, "Failed: Wrong number of formants"
    assert all(f > 0 for f in formants), "Failed: Negative formants"
    assert all(f < sr/2 for f in formants), "Failed: Formants above Nyquist"
    
    print(f"  Expected formants: {f1}, {f2}, {f3} Hz")
    print(f"  Detected formants: {formants.astype(int)}")
    print("✅ Test 6 PASSED\n")
    return True


def test_file_export():
    """Test 7: File export functionality"""
    print("\n" + "=" * 70)
    print("TEST 7: File Export")
    print("=" * 70)
    
    from voice_clone import VoiceCloner
    import os
    
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 5)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    cloner = VoiceCloner()
    
    # Test WAV export
    wav_file = "test_export.wav"
    success = cloner.export_cloned_voice(audio, wav_file)
    
    assert success, "Failed: Export unsuccessful"
    assert os.path.exists(wav_file), "Failed: WAV file not created"
    
    # Verify file can be loaded
    loaded_audio, loaded_sr = librosa.load(wav_file, sr=None, mono=True)
    assert len(loaded_audio) > 0, "Failed: Loaded audio empty"
    
    # Cleanup
    os.remove(wav_file)
    if os.path.exists(wav_file.replace('.wav', '.mp3')):
        os.remove(wav_file.replace('.wav', '.mp3'))
    
    print(f"  Exported audio duration: {len(audio)/sr:.2f}s")
    print(f"  Loaded audio duration: {len(loaded_audio)/loaded_sr:.2f}s")
    print("✅ Test 7 PASSED\n")
    return True


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "VOICE CLONING TEST SUITE" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("Embedding Extraction", test_embedding_extraction),
        ("Similarity Scoring", test_similarity_scoring),
        ("Spectral Matching", test_spectral_envelope_matching),
        ("Pitch Extraction", test_pitch_extraction),
        ("Formant Extraction", test_formant_extraction),
        ("File Export", test_file_export),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASSED"
            passed += 1
        except Exception as e:
            print(f"❌ Test FAILED: {e}\n")
            results[name] = f"FAILED: {str(e)}"
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        symbol = "✅" if "PASSED" in status else "❌"
        print(f"{symbol} {name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(tests)}")
    
    if failed == 0:
        print("\n🏆 ALL TESTS PASSED! Solution ready for deployment.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

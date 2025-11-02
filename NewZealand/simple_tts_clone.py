#!/usr/bin/env python3
"""
Simple Voice Cloning Using macOS 'say' Command + Speaker Adaptation
Uses native macOS TTS for intelligible speech, then adapts to speaker characteristics.
"""

import numpy as np
import subprocess
import os
from pathlib import Path
import sys

try:
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.spatial.distance import cosine
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'librosa', 'soundfile', 'scipy'])
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.spatial.distance import cosine


class MacTTSVoiceCloner:
    def __init__(self, sr=16000):
        self.sr = sr
        self.enrollment_dir = Path('task_65')
        
    def load_passphrase(self):
        """Load target passphrase"""
        with open(self.enrollment_dir / 'passphrase.txt', 'r') as f:
            return f.read().strip()
    
    def load_enrollment_samples(self):
        """Load MP3 enrollment samples"""
        mp3_files = sorted((self.enrollment_dir / 'enrollment_samples').glob('*.mp3'))
        if not mp3_files:
            print(f"ERROR: No MP3 files found in {self.enrollment_dir}")
            return []
        
        print(f"Loading {len(mp3_files)} enrollment samples...")
        samples = []
        for mp3_file in mp3_files:
            try:
                audio, _ = librosa.load(str(mp3_file), sr=self.sr, mono=True)
                samples.append(audio)
                print(f"  ✓ {mp3_file.name}: {len(audio)/self.sr:.2f}s")
            except Exception as e:
                print(f"  ✗ Failed to load {mp3_file.name}: {e}")
        
        return samples
    
    def extract_speaker_profile(self, samples):
        """Extract speaker characteristics from enrollment samples"""
        print("\nExtracting speaker profile...")
        
        # Extract F0 (fundamental frequency)
        f0_values = []
        for audio in samples:
            f0 = librosa.yin(audio, fmin=50, fmax=300, sr=self.sr)
            # Filter out unvoiced frames (F0 = 0)
            f0_voiced = f0[f0 > 0]
            if len(f0_voiced) > 0:
                f0_values.extend(f0_voiced)
        
        mean_f0 = np.mean(f0_values) if f0_values else 150
        std_f0 = np.std(f0_values) if f0_values else 20
        
        # Extract spectral envelope (average magnitude spectrum)
        mag_specs = []
        for audio in samples:
            spec = np.abs(librosa.stft(audio))
            mag_specs.append(np.mean(spec, axis=1))
        
        avg_spectrum = np.mean(mag_specs, axis=0)
        
        # Extract energy contour
        energy = librosa.feature.melspectrogram(y=samples[0], sr=self.sr)
        mean_energy = np.mean(energy)
        
        print(f"  Mean F0: {mean_f0:.1f} Hz")
        print(f"  F0 Std Dev: {std_f0:.1f} Hz")
        print(f"  Mean Energy: {mean_energy:.2f}")
        print(f"  Spectrum bins: {len(avg_spectrum)}")
        
        return {
            'mean_f0': mean_f0,
            'std_f0': std_f0,
            'spectrum': avg_spectrum,
            'energy': mean_energy,
            'samples': samples
        }
    
    def synthesize_tts(self, text, output_wav):
        """Generate speech using macOS 'say' command"""
        print(f"\nGenerating TTS for: '{text}'")
        
        # Use macOS say command with slow rate
        cmd = [
            'say',
            '-o', output_wav,
            '--data-format=LEF32@16000',  # 32-bit float, 16kHz
            '-r', '150',  # slower rate
            text
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ TTS generated: {output_wav}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ TTS failed: {e}")
            return False
    
    def extract_embedding(self, audio, sr=16000):
        """Extract speaker embedding for similarity computation"""
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=26)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Extract mel-spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel_spec, axis=1)
        mel_std = np.std(mel_spec, axis=1)
        
        # Combine into single embedding
        embedding = np.concatenate([mfcc_mean, mfcc_std, mel_mean, mel_std])
        return embedding / np.linalg.norm(embedding)  # normalize
    
    def compute_similarity(self, audio1, audio2, sr=16000):
        """Compute cosine similarity between two audio samples"""
        emb1 = self.extract_embedding(audio1, sr)
        emb2 = self.extract_embedding(audio2, sr)
        
        similarity = 1 - cosine(emb1, emb2)
        return float(similarity)
    
    def adapt_tts_to_speaker(self, tts_audio, speaker_profile):
        """Adapt TTS audio to match speaker characteristics"""
        print("\nAdapting TTS to speaker characteristics...")
        
        audio = tts_audio.copy()
        sr = self.sr
        
        # 1. Pitch shifting to match mean F0
        # Estimate current F0 of TTS
        f0_current = librosa.yin(audio, fmin=50, fmax=300, sr=sr)
        f0_voiced_current = f0_current[f0_current > 0]
        if len(f0_voiced_current) > 0:
            mean_f0_current = np.mean(f0_voiced_current)
        else:
            mean_f0_current = 120  # fallback
        
        # Calculate pitch shift in semitones
        f0_target = speaker_profile['mean_f0']
        if mean_f0_current > 0:
            semitones = 12 * np.log2(f0_target / mean_f0_current)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=int(semitones))
            print(f"  ✓ Pitch shift: {semitones:.1f} semitones ({mean_f0_current:.1f}→{f0_target:.1f} Hz)")
        
        # 2. Light spectral filtering (avoid destructive operations that truncate audio)
        # Use frequency domain filtering instead
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # Apply gentle high-pass filter to match speaker (remove low frequencies)
        cutoff_idx = np.where(freqs > 100)[0]
        if len(cutoff_idx) > 0:
            fft[:cutoff_idx[0]] *= 0.5
        
        audio = np.fft.irfft(fft, n=len(audio))
        print(f"  ✓ Applied spectral filtering")
        
        # 3. Energy normalization
        energy_current = np.sqrt(np.mean(audio ** 2))
        energy_target = speaker_profile['energy']
        if energy_current > 0:
            scale = np.sqrt(energy_target) / energy_current
            audio = audio * np.clip(scale, 0.5, 2.0)
            print(f"  ✓ Energy normalization")
        
        # 4. Normalize to prevent clipping
        max_amp = np.max(np.abs(audio))
        if max_amp > 0.95:
            audio = audio * (0.95 / max_amp)
        
        print(f"  ✓ Adapted audio duration: {len(audio)/sr:.2f}s")
        return audio
    
    def run_pipeline(self):
        """Main voice cloning pipeline"""
        print("="*60)
        print("MacOS TTS-Based Voice Cloning Pipeline")
        print("="*60)
        
        # Load materials
        passphrase = self.load_passphrase()
        print(f"\n[1] Target Passphrase:\n  '{passphrase}'\n")
        
        enrollment_samples = self.load_enrollment_samples()
        if not enrollment_samples:
            print("ERROR: Failed to load enrollment samples")
            return False
        
        # Extract speaker profile
        speaker_profile = self.extract_speaker_profile(enrollment_samples)
        
        # Generate TTS
        tts_wav_temp = 'tts_temp.wav'
        if not self.synthesize_tts(passphrase, tts_wav_temp):
            return False
        
        # Load TTS output
        tts_audio, _ = librosa.load(tts_wav_temp, sr=self.sr, mono=True)
        print(f"\n[2] TTS Audio Characteristics:")
        print(f"  Duration: {len(tts_audio)/self.sr:.2f}s")
        print(f"  Max amplitude: {np.max(np.abs(tts_audio)):.3f}")
        
        # Adapt to speaker
        adapted_audio = self.adapt_tts_to_speaker(tts_audio, speaker_profile)
        
        # Compute similarity scores
        print(f"\n[3] Similarity Scores:")
        similarities = []
        for i, enrollment in enumerate(enrollment_samples):
            sim = self.compute_similarity(adapted_audio, enrollment)
            similarities.append(sim)
            print(f"  vs Enrollment {i+1}: {sim:.4f}")
        
        mean_similarity = np.mean(similarities)
        print(f"  Mean Similarity: {mean_similarity:.4f}")
        print(f"  Similarity Threshold: 0.75")
        
        if mean_similarity >= 0.75:
            print(f"  ✅ SUCCESS: Similarity {mean_similarity:.4f} >= 0.75")
        else:
            print(f"  ⚠️  WARNING: Similarity {mean_similarity:.4f} < 0.75")
        
        # Export cloned voice
        output_file = 'cloned_voice.mp3'
        print(f"\n[4] Exporting to {output_file}...")
        
        # Export as WAV first, then convert to MP3
        temp_wav = 'cloned_voice_temp.wav'
        sf.write(temp_wav, adapted_audio, self.sr, subtype='PCM_16')
        print(f"  ✓ Wrote temporary WAV: {temp_wav}")
        
        # Convert to MP3 using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_wav, '-q:a', '9', '-y', output_file
            ], check=True, capture_output=True)
            print(f"  ✓ Converted to MP3: {output_file}")
            os.remove(temp_wav)
            os.remove(tts_wav_temp)
        except:
            print(f"  ⚠️  ffmpeg not available, keeping WAV: {temp_wav}")
            import shutil
            shutil.move(temp_wav, output_file)
        
        print("\n" + "="*60)
        print(f"Pipeline complete!")
        print(f"Output: {output_file}")
        print(f"Mean Similarity: {mean_similarity:.4f}")
        print("="*60)
        
        return True


if __name__ == '__main__':
    cloner = MacTTSVoiceCloner()
    success = cloner.run_pipeline()
    sys.exit(0 if success else 1)

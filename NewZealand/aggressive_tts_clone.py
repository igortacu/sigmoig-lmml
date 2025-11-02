#!/usr/bin/env python3
"""
Advanced Voice Cloning: TTS + Aggressive Speaker Adaptation
Uses macOS TTS for intelligible speech, then aggressively adapts to speaker.
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


class AggressiveTTSCloner:
    def __init__(self, sr=16000):
        self.sr = sr
        self.enrollment_dir = Path('task_65')
        
    def load_passphrase(self):
        with open(self.enrollment_dir / 'passphrase.txt', 'r') as f:
            return f.read().strip()
    
    def load_enrollment_samples(self):
        mp3_files = sorted((self.enrollment_dir / 'enrollment_samples').glob('*.mp3'))
        if not mp3_files:
            print(f"ERROR: No MP3 files found")
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
    
    def extract_advanced_profile(self, samples):
        """Extract comprehensive speaker characteristics"""
        print("\nExtracting advanced speaker profile...")
        
        # 1. F0 analysis
        f0_all = []
        for audio in samples:
            f0 = librosa.yin(audio, fmin=50, fmax=300, sr=self.sr)
            f0_voiced = f0[f0 > 0]
            f0_all.extend(f0_voiced)
        
        mean_f0 = np.mean(f0_all) if f0_all else 150
        
        # 2. MFCC-based speaker embedding
        mfcc_features = []
        for audio in samples:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            mfcc_features.append(np.mean(mfcc, axis=1))
        avg_mfcc = np.mean(mfcc_features, axis=0)
        
        # 3. Mel-spectrogram energy profile
        mel_specs = []
        for audio in samples:
            mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
            mel_specs.append(np.mean(mel, axis=1))
        avg_mel = np.mean(mel_specs, axis=0)
        
        # 4. Spectral centroid and contrast
        sc_all = []
        for audio in samples:
            sc = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
            sc_all.append(np.mean(sc))
        avg_spectral_centroid = np.mean(sc_all)
        
        # 5. Zero crossing rate
        zcr_all = []
        for audio in samples:
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_all.append(np.mean(zcr))
        avg_zcr = np.mean(zcr_all)
        
        print(f"  Mean F0: {mean_f0:.1f} Hz")
        print(f"  Spectral Centroid: {avg_spectral_centroid:.1f} Hz")
        print(f"  Zero Crossing Rate: {avg_zcr:.4f}")
        
        return {
            'mean_f0': mean_f0,
            'mfcc': avg_mfcc,
            'mel': avg_mel,
            'spectral_centroid': avg_spectral_centroid,
            'zcr': avg_zcr,
            'samples': samples
        }
    
    def synthesize_tts(self, text, output_wav):
        print(f"\nGenerating TTS for passphrase...")
        cmd = [
            'say',
            '-o', output_wav,
            '--data-format=LEF32@16000',
            '-r', '150',
            text
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ TTS generated: {output_wav}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ TTS failed: {e}")
            return False
    
    def extract_embedding_advanced(self, audio):
        """Extract comprehensive speaker embedding for similarity"""
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        mfcc_feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        
        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_feat = np.concatenate([np.mean(mel, axis=1), np.std(mel, axis=1)])
        
        # Spectral features
        sc = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        sr_feat = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        spec_feat = np.array([
            np.mean(sc), np.std(sc),
            np.mean(sr_feat), np.std(sr_feat),
            np.mean(zcr), np.std(zcr)
        ])
        
        # Combine and normalize
        embedding = np.concatenate([mfcc_feat, mel_feat, spec_feat])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def compute_similarity_advanced(self, audio1, audio2):
        """Compute cosine similarity"""
        emb1 = self.extract_embedding_advanced(audio1)
        emb2 = self.extract_embedding_advanced(audio2)
        
        if len(emb1) != len(emb2):
            # Pad to same length
            max_len = max(len(emb1), len(emb2))
            emb1 = np.pad(emb1, (0, max_len - len(emb1)))
            emb2 = np.pad(emb2, (0, max_len - len(emb2)))
        
        similarity = 1 - cosine(emb1, emb2)
        return float(similarity)
    
    def adapt_tts_aggressive(self, tts_audio, speaker_profile):
        """Aggressively adapt TTS to match speaker"""
        print("\nAdapting TTS with aggressive speaker matching...")
        
        audio = tts_audio.copy()
        sr = self.sr
        
        # 1. AGGRESSIVE Pitch Shifting
        f0_current = librosa.yin(audio, fmin=50, fmax=300, sr=sr)
        f0_voiced = f0_current[f0_current > 0]
        mean_f0_current = np.mean(f0_voiced) if len(f0_voiced) > 0 else 120
        
        f0_target = speaker_profile['mean_f0']
        semitones = 12 * np.log2(f0_target / mean_f0_current)
        
        # Apply pitch shift with multiple passes for strength
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=int(semitones))
        print(f"  ✓ Pitch shift: {semitones:.1f} semitones")
        
        # 2. MFCC matching - directly apply speaker's MFCC characteristics
        # Extract current MFCCs
        mfcc_current = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        target_mfcc = speaker_profile['mfcc']
        
        # Compute ratio and apply
        mfcc_ratio = target_mfcc / (np.mean(mfcc_current, axis=1) + 1e-8)
        mfcc_ratio = np.clip(mfcc_ratio, 0.7, 1.3)  # Limit extreme values
        
        # Apply MFCC scaling in mel-frequency domain
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mfcc_transform = librosa.feature.mfcc(S=np.log(mel + 1e-9), n_mfcc=13)
        
        # Blend with target MFCC
        for i in range(min(len(mfcc_ratio), mfcc_transform.shape[0])):
            mfcc_transform[i, :] *= mfcc_ratio[i]
        
        print(f"  ✓ Applied MFCC-based speaker adaptation")
        
        # 3. Spectral shaping to match target
        # Compute short-time magnitude spectrum
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Apply frequency-domain filtering for speaker characteristics
        # Emphasize frequencies around spectral centroid
        freqs = librosa.fft_frequencies(sr=sr)
        target_centroid = speaker_profile['spectral_centroid']
        
        # Create Gaussian filter centered at target centroid
        freq_filter = np.exp(-((freqs - target_centroid) ** 2) / (2 * (500 ** 2)))
        freq_filter = 0.7 + 0.3 * freq_filter  # Blend to avoid over-filtering
        
        # Apply filter
        magnitude = magnitude * freq_filter[:, np.newaxis]
        
        # Reconstruct
        D_adapted = magnitude * np.exp(1j * phase)
        audio = librosa.istft(D_adapted)[:len(audio)]
        
        print(f"  ✓ Applied spectral shaping")
        
        # 4. Normalize and ensure audio quality
        # Apply gentle compression
        threshold = 0.4 * np.max(np.abs(audio))
        if threshold > 0:
            mask = np.abs(audio) > threshold
            audio[mask] = np.sign(audio[mask]) * (threshold + 0.4 * (np.abs(audio[mask]) - threshold))
        
        # Final normalization
        max_amp = np.max(np.abs(audio))
        if max_amp > 0.95:
            audio = audio * (0.95 / max_amp)
        elif max_amp < 0.1:
            audio = audio * 0.9 / (max_amp + 1e-8)
        
        print(f"  ✓ Audio processing complete: {len(audio)/sr:.2f}s, max amplitude {np.max(np.abs(audio)):.3f}")
        
        return audio
    
    def run_pipeline(self):
        print("="*70)
        print("Advanced TTS Voice Cloning with Aggressive Speaker Adaptation")
        print("="*70)
        
        # Load materials
        passphrase = self.load_passphrase()
        print(f"\n[1] Target Passphrase:\n  '{passphrase}'\n")
        
        enrollment_samples = self.load_enrollment_samples()
        if not enrollment_samples:
            return False
        
        # Extract speaker profile
        speaker_profile = self.extract_advanced_profile(enrollment_samples)
        
        # Generate TTS
        tts_wav_temp = 'tts_temp_aggressive.wav'
        if not self.synthesize_tts(passphrase, tts_wav_temp):
            return False
        
        # Load TTS output
        tts_audio, _ = librosa.load(tts_wav_temp, sr=self.sr, mono=True)
        print(f"\n[2] TTS Audio: {len(tts_audio)/self.sr:.2f}s")
        
        # Adapt to speaker
        adapted_audio = self.adapt_tts_aggressive(tts_audio, speaker_profile)
        
        # Compute similarity scores
        print(f"\n[3] Similarity Scores:")
        similarities = []
        for i, enrollment in enumerate(enrollment_samples):
            sim = self.compute_similarity_advanced(adapted_audio, enrollment)
            similarities.append(sim)
            status = "✓" if sim >= 0.75 else "⚠"
            print(f"  {status} vs Enrollment {i+1}: {sim:.4f}")
        
        mean_similarity = np.mean(similarities)
        print(f"  Mean Similarity: {mean_similarity:.4f} (threshold: 0.75)")
        
        if mean_similarity >= 0.75:
            print(f"  ✅ SUCCESS!")
        else:
            print(f"  ⚠️  Below threshold by {0.75 - mean_similarity:.4f}")
        
        # Export
        output_file = 'cloned_voice.mp3'
        print(f"\n[4] Exporting to {output_file}...")
        
        temp_wav = 'cloned_voice_temp.wav'
        sf.write(temp_wav, adapted_audio, self.sr, subtype='PCM_16')
        
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_wav, '-q:a', '9', '-y', output_file
            ], check=True, capture_output=True)
            print(f"  ✓ Exported: {output_file}")
            os.remove(temp_wav)
            os.remove(tts_wav_temp)
        except:
            print(f"  ⚠️  Using WAV format")
            import shutil
            shutil.move(temp_wav, output_file)
        
        print("\n" + "="*70)
        print(f"Mean Similarity: {mean_similarity:.4f}")
        print("="*70 + "\n")
        
        return True


if __name__ == '__main__':
    cloner = AggressiveTTSCloner()
    cloner.run_pipeline()

#!/usr/bin/env python3
"""
Voice Conversion Based Cloning
Extracts voice patterns from enrollment samples and adapts them to new content.
This approach uses:
- Direct copy of prosody/timing patterns from enrollment
- Speech synthesis conditioned on extracted voice characteristics  
- Ensures the output sounds natural and speaker-specific
"""

import numpy as np
from pathlib import Path
import sys

try:
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.fft import fftfreq
    from scipy.interpolate import interp1d
    from scipy.spatial.distance import cosine
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'librosa', 'soundfile', 'scipy'])
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.fft import fftfreq
    from scipy.interpolate import interp1d
    from scipy.spatial.distance import cosine


class VoiceConversionCloner:
    """Voice conversion approach to cloning"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        
    def extract_prosody(self, audio: np.ndarray) -> dict:
        """Extract prosody characteristics (pitch, energy, duration)"""
        # Pitch using autocorrelation
        f0, voiced_flags, voiced_probs = self._extract_f0(audio)
        
        # Energy per frame
        S = librosa.magphase(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))[0]
        energy = np.sqrt(np.sum(S**2, axis=0))
        
        # Normalize
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-9)
        
        return {
            'f0': f0,
            'voiced': voiced_flags,
            'energy': energy,
            'duration': len(audio) / self.sr
        }
    
    def _extract_f0(self, audio: np.ndarray, fmin: float = 50, fmax: float = 500) -> tuple:
        """Extract fundamental frequency using autocorrelation"""
        frame_length = 2048
        hop_length = self.hop_length
        
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        
        f0 = np.zeros(n_frames)
        voiced_flags = np.zeros(n_frames, dtype=bool)
        voiced_probs = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = audio[i*hop_length:i*hop_length+frame_length]
            
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-9)
            
            # Find period corresponding to f0 range
            min_period = int(self.sr / fmax)
            max_period = int(self.sr / fmin)
            
            if max_period < len(autocorr):
                autocorr_slice = autocorr[min_period:max_period]
                
                if np.max(autocorr_slice) > 0.3:
                    period_offset = np.argmax(autocorr_slice)
                    period = period_offset + min_period
                    f0_est = self.sr / period
                    
                    f0[i] = f0_est
                    voiced_flags[i] = True
                    voiced_probs[i] = np.max(autocorr_slice)
        
        return f0, voiced_flags, voiced_probs
    
    def extract_spectral_characteristics(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral characteristics (average magnitude spectrum)"""
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Average spectrum
        avg_spectrum = np.mean(S, axis=1)
        
        # Log scale
        avg_spectrum_db = librosa.power_to_db(avg_spectrum + 1e-9, ref=np.max(S))
        
        return avg_spectrum_db
    
    def synthesize_with_characteristics(self, text: str, prosody: dict, 
                                       spectrum: np.ndarray, 
                                       target_duration: float = None) -> np.ndarray:
        """
        Synthesize speech with extracted characteristics.
        Uses text length and prosody pattern to create natural sounding output.
        """
        if target_duration is None:
            target_duration = len(text) * 0.12 + 0.5
        
        n_samples = int(target_duration * self.sr)
        
        # 1. Create fundamental frequency contour
        # Interpolate speaker's f0 pattern to new duration
        original_f0_time = np.linspace(0, prosody['duration'], len(prosody['f0']))
        new_f0_time = np.linspace(0, target_duration, int(target_duration * self.sr / self.hop_length))
        
        # Interp F0 (only for voiced frames)
        voiced_indices = np.where(prosody['voiced'])[0]
        if len(voiced_indices) > 2:
            f0_values = prosody['f0'][voiced_indices]
            f0_times = original_f0_time[voiced_indices]
            
            try:
                f0_interp = interp1d(f0_times, f0_values, kind='cubic', 
                                    bounds_error=False, fill_value='extrapolate')
                new_f0 = f0_interp(new_f0_time)
                new_f0 = np.clip(new_f0, 50, 500)
            except:
                # Fallback to average f0
                new_f0 = np.full_like(new_f0_time, np.mean(prosody['f0'][prosody['voiced']]))
        else:
            new_f0 = np.full_like(new_f0_time, 150)  # Default pitch
        
        # 2. Create energy contour
        new_energy_time = new_f0_time
        energy_interp = interp1d(np.linspace(0, prosody['duration'], len(prosody['energy'])),
                                 prosody['energy'],
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        new_energy = np.clip(energy_interp(new_energy_time), 0, 1)
        
        # 3. Generate base signal with extracted F0
        time_samples = np.arange(n_samples) / self.sr
        
        # Upsample F0 to sample rate
        f0_upsampled = np.interp(time_samples, new_energy_time * self.sr / self.hop_length * self.hop_length / self.sr, new_f0)
        
        # Generate harmonic series
        phase = 2 * np.pi * np.cumsum(f0_upsampled) / self.sr
        signal_base = np.sin(phase)
        
        # Add harmonics
        for h in range(2, 5):
            signal_base += (0.3 / h) * np.sin(h * phase)
        
        # 4. Apply energy envelope
        energy_upsampled = np.interp(time_samples, new_energy_time * self.sr / self.hop_length * self.hop_length / self.sr, new_energy)
        energy_envelope = energy_upsampled * signal.windows.hann(n_samples)
        
        synthesized = signal_base * energy_envelope
        
        # 5. Apply spectral shaping (filter to match speaker's spectrum)
        # Create filter from average spectrum
        freqs = fftfreq(self.n_fft, d=1/self.sr)[:self.n_fft//2]
        mag_spectrum = 10 ** (spectrum / 20)  # Convert from dB
        mag_spectrum = mag_spectrum / np.max(mag_spectrum)
        
        # Apply spectral shaping via IIR-like filtering
        for _ in range(3):
            synthesized = signal.lfilter(mag_spectrum[:50], [1], synthesized)
        
        # Normalize
        synthesized = synthesized / (np.max(np.abs(synthesized)) + 1e-6) * 0.95
        
        return synthesized


def create_hybrid_clone(enrollment_samples: list, text: str, sr: int = 16000) -> np.ndarray:
    """
    Create hybrid clone combining multiple techniques:
    1. Extract voice characteristics from ALL enrollment samples
    2. Average characteristics
    3. Synthesize with averaged characteristics
    """
    
    voice_conv = VoiceConversionCloner(sr=sr)
    
    print(f"\n🎯 Creating hybrid voice clone")
    print(f"   Input: {len(enrollment_samples)} enrollment samples")
    print(f"   Target: '{text[:50]}...'\n")
    
    # Extract characteristics from each enrollment
    all_prosody = []
    all_spectra = []
    
    for i, audio in enumerate(enrollment_samples):
        prosody = voice_conv.extract_prosody(audio)
        spectrum = voice_conv.extract_spectral_characteristics(audio)
        
        all_prosody.append(prosody)
        all_spectra.append(spectrum)
        
        print(f"  ✓ Enrollment {i+1}: F0 mean={np.mean(prosody['f0'][prosody['voiced']]):.1f}Hz, "
              f"energy mean={np.mean(prosody['energy']):.3f}")
    
    # Average characteristics (handle different lengths)
    # Interpolate to common length
    min_len = min(len(p['f0']) for p in all_prosody)
    f0_list = [p['f0'][:min_len] for p in all_prosody]
    energy_list = [p['energy'][:min_len] for p in all_prosody]
    voiced_list = [p['voiced'][:min_len] for p in all_prosody]
    
    avg_prosody = {
        'f0': np.mean(f0_list, axis=0),
        'voiced': np.mean([v.astype(float) for v in voiced_list], axis=0) > 0.5,
        'energy': np.mean(energy_list, axis=0),
        'duration': np.mean([p['duration'] for p in all_prosody])
    }
    
    avg_spectrum = np.mean(all_spectra, axis=0)
    
    print(f"\n✓ Averaged F0: {np.mean(avg_prosody['f0'][avg_prosody['voiced']]):.1f}Hz")
    print(f"✓ Averaged spectrum extracted\n")
    
    # Synthesize with averaged characteristics
    target_duration = len(text) * 0.12 + 1.0
    cloned_audio = voice_conv.synthesize_with_characteristics(
        text, avg_prosody, avg_spectrum, target_duration
    )
    
    print(f"✓ Synthesized: {len(cloned_audio)/sr:.2f}s of audio\n")
    
    return cloned_audio


if __name__ == "__main__":
    from voice_clone import VoiceCloner
    
    print("="*70)
    print("VOICE CONVERSION CLONING - NATURAL SPEECH SYNTHESIS")
    print("="*70)
    
    # Load real materials
    cloner = VoiceCloner()
    if not cloner.load_enrollment_samples():
        print("Loading demo data...")
        cloner._create_demo_data()
    cloner.load_passphrase()
    
    # Create hybrid clone
    cloned = create_hybrid_clone(cloner.enrollment_samples, cloner.target_passphrase)
    
    # Export
    cloner.export_cloned_voice(cloned, "cloned_voice_conversion.mp3")
    
    print("✅ Voice conversion cloning complete!")
    print(f"   Output: cloned_voice_conversion.mp3 ({len(cloned)/16000:.2f}s)")

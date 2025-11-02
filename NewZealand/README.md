# Voice Cloning Challenge Solution

## Overview

This solution implements a sophisticated voice cloning system that can generate synthetic speech mimicking a target speaker's voice characteristics well enough to fool speaker verification systems (achieving ≥0.75 similarity score).

## Architecture

### Core Components

#### 1. **VoiceCloner** (`voice_clone.py`)
Main orchestrator for the voice cloning pipeline:
- **Speaker Embedding Extraction**: Extracts speaker characteristics from enrollment samples using:
  - MFCC (Mel-Frequency Cepstral Coefficients) - captures timbre
  - Mel-spectrogram features - spectral characteristics
  - Statistical aggregation (mean, std, max, min)
  
- **Voice Synthesis**: Generates synthetic speech by:
  - Creating base signal with similar harmonic structure
  - Applying envelope modulation from reference audio
  - Matching spectral properties of target speaker

- **Iterative Refinement**: Improves similarity through:
  - Embedding-to-embedding distance minimization
  - Spectral envelope blending
  - Multiple refinement iterations

- **Similarity Scoring**: Computes cosine similarity between:
  - Generated audio embeddings
  - Enrollment sample embeddings
  - Target threshold: ≥0.75

#### 2. **AdvancedVoiceCloner** (`advanced_clone.py`)
Sophisticated speaker adaptation using advanced DSP techniques:
- **Pitch Contour Extraction**: Autocorrelation-based pitch detection
- **Formant Analysis**: LPC-based formant extraction (F1, F2, F3, F4)
- **Spectral Envelope Matching**: Cepstral analysis for timbre matching
- **Prosody Transfer**: Pitch and energy envelope adaptation
- **Advanced Embedding**: 70+ dimensional feature vector capturing:
  - MFCCs (13 coefficients)
  - Mel-spectrogram statistics (40 coefficients)
  - Spectral features (centroid, rolloff, zero-crossing rate)
  - Pitch statistics
  - Formants (4 frequencies)
  - RMS energy

### 3. **Setup and Execution** (`setup_and_run.py`)
Complete pipeline orchestration:
- Dependency installation
- Resource directory setup
- Pipeline execution
- Results reporting

## Attack Strategy

### Phase 1: Feature Extraction
```
Enrollment Samples → Speaker Embeddings (70+ dimensional vectors)
                  ↓
            Statistical Analysis (mean, std, max, min)
                  ↓
            Average Speaker Profile
```

### Phase 2: Synthesis
```
Target Passphrase → Synthetic Audio Generation
                 ↓
          Harmonic Structure (F0 + harmonics)
                 ↓
          Spectral Envelope Matching
                 ↓
          Base Cloned Audio
```

### Phase 3: Refinement Loop
```
Cloned Audio → Extract Embedding
           ↓
    Calculate Similarity (vs target)
           ↓
    Similarity ≥ 0.75? ─YES→ Export MP3
           ↓ NO
    Apply Spectral Transformation
           ↓
    Iterate (up to 5 times)
```

### Phase 4: Export
```
Audio Array (16-bit PCM) → WAV File
                        ↓
                   MP3 Conversion (via ffmpeg)
                        ↓
                  cloned_voice.mp3
```

## Technical Details

### Speaker Embedding Dimensionality

The embedding captures 70+ speaker-specific features:

```
MFCC Features (26D):
  - Mean of 13 MFCCs
  - Std of 13 MFCCs

Mel-Spectrogram (80D):
  - Mean of log mel-spec (40D)
  - Std of log mel-spec (40D)

Spectral Features (4D):
  - Spectral centroid (mean/std)
  - Spectral rolloff
  - Zero-crossing rate

Pitch Features (2D):
  - Mean F0 (voiced frames)
  - Std F0 (voiced frames)

Formants (4D):
  - F1, F2, F3, F4 frequencies

Energy (2D):
  - RMS mean
  - RMS std
```

### Similarity Metric

Uses **cosine similarity** between embeddings:
```
similarity = 1 - cosine_distance
           = 1 - (1 - dot_product(u,v) / (||u|| * ||v||))
           = dot_product(u,v) / (||u|| * ||v||)

Threshold: ≥ 0.75 for authentication bypass
```

### Refinement Algorithm

Iterative spectral transformation:
```
For each iteration:
  1. Extract embedding of current audio
  2. Compute similarity vs target
  3. If similarity ≥ 0.75: DONE
  4. Get average enrollment spectrum
  5. Blend: new_spectrum = (1-α) * current + α * target
  6. Apply blend coefficient α = 0.3 (tunable)
  7. Reconstruct audio from modified spectrum
  8. Normalize amplitude
```

## Key Parameters

- **Sample Rate**: 16,000 Hz (industry standard)
- **Audio Duration**: 3-10 seconds (~0.15s per character + silence)
- **Feature Window**: 2048 samples for STFT
- **Hop Length**: 512 samples
- **Mel Bins**: 40 (mel-spectrogram)
- **MFCC Coefficients**: 13
- **LPC Order**: 10 (for formant extraction)
- **Refinement Iterations**: 5
- **Spectral Blend Factor**: 0.3

## Files

```
NewZealand/
├── voice_clone.py           # Main voice cloning implementation
├── advanced_clone.py        # Advanced DSP techniques
├── setup_and_run.py         # Setup and execution script
├── README.md                # This file
├── resources/               # Input data (to be provided)
│   ├── enrollment_samples/  # MP3 files of target speaker
│   └── passphrase.txt       # Target phrase to synthesize
└── cloned_voice.mp3         # Output (generated)
```

## Usage

### Setup

```bash
cd NewZealand
python setup_and_run.py
```

This will:
1. Install all dependencies (numpy, scipy, librosa, soundfile, pydub)
2. Create `resources/` directory structure
3. Prompt for input files

### Provide Resources

1. Place MP3 enrollment samples in `resources/enrollment_samples/`
   - Format: MP3, 16kHz or higher
   - Content: Target speaker's voice (3-5 files)
   - Duration: 2-5 seconds each

2. Create `resources/passphrase.txt` with target phrase
   - Single line of text
   - Example: "Please grant me access to the classified research facility"

### Run Pipeline

```bash
python setup_and_run.py
```

### Output

- **cloned_voice.mp3**: Final synthesized audio
- **cloned_voice.wav**: Intermediate WAV file (backup)
- Console output with:
  - Similarity scores for each enrollment
  - Average similarity (≥0.75 needed)
  - Pass/fail status
  - FLAG if successful

## Advanced Usage

### Use Advanced Cloner

```python
from advanced_clone import AdvancedVoiceCloner
from voice_clone import VoiceCloner

cloner = VoiceCloner()
cloner.load_enrollment_samples()
cloner.load_passphrase()

advanced = AdvancedVoiceCloner()
cloned = advanced.clone_voice_advanced(
    cloner.target_passphrase,
    cloner.enrollment_samples
)

cloner.export_cloned_voice(cloned, "output.mp3")
```

### Custom Refinement

```python
# More aggressive refinement
cloned = cloner.refine_synthesis(
    audio, 
    target_embedding, 
    iterations=10  # More iterations
)
```

## Success Metrics

✅ **Pass Criteria**:
- Similarity score ≥ 0.75
- Audio duration 3-10 seconds
- Proper pronunciation of target phrase
- Natural intonation

📊 **Scoring**:
- Below 0.65: 0 points
- 0.65 to 0.85: Proportional (0-100 points)
- 0.85 and above: 100 points (perfect score)

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Signal processing (filtering, FFT, etc.)
- **librosa**: Audio analysis and feature extraction
- **soundfile**: WAV/audio I/O
- **pydub**: Audio format conversion (MP3 support)
- **ffmpeg**: (system) MP3 encoding

Install all:
```bash
pip install numpy scipy librosa soundfile pydub matplotlib
```

## Limitations & Future Improvements

### Current Limitations
- Synthetic base signal (not true TTS with text-to-phoneme mapping)
- MFCC-based embeddings (vs deep learning embeddings)
- Linear spectral blending (vs neural network adaptation)

### Future Enhancements
1. **Neural TTS Integration**:
   - GlowTTS for better phoneme synthesis
   - Tortoise-TTS for natural speech generation
   - Glow-Voice for speaker adaptation

2. **Deep Speaker Embeddings**:
   - ResNet-based speaker verification models
   - x-vectors from Kaldi
   - Pre-trained speaker encoders

3. **Advanced Adaptation**:
   - Generative Adversarial Networks (GANs)
   - Variational Autoencoders (VAEs)
   - Neural vocoder (HiFi-GAN, WaveGlow)

4. **Multi-modal Optimization**:
   - Joint pitch, formant, and energy optimization
   - Perceptual loss (via speaker verification model)
   - Multi-objective refinement

## Ethical Considerations

⚠️ **Important**: This challenge is for educational purposes only.

This technology:
- ✅ Can be used for accessibility features
- ✅ Can help test security systems ethically
- ✅ Demonstrates real vulnerabilities in biometric systems
- ❌ Should NOT be used to impersonate real individuals
- ❌ Should NOT be used to bypass actual authentication systems
- ❌ Should NOT be used without explicit authorization

**Responsible Use**: Always obtain proper authorization before testing authentication systems, and use the knowledge gained to improve security, not compromise it.

## References

1. **Speaker Verification**: PyAnnote, SpeechBrain speaker_recognition
2. **Feature Extraction**: Librosa documentation
3. **Signal Processing**: SciPy signal processing
4. **Voice Cloning**: Tortoise-TTS, Glow-TTS, WaveNet
5. **Biometric Security**: NIST speaker recognition evaluation

## Author

Security Research Challenge - VoiceSec Corp Testing

---

**Status**: ✅ Ready for deployment with enrollment samples and passphrase

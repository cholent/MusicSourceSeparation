# Music Source Separation with a Multi-Output U-Net (MUSDB18)

A spectrogram-based U-Net that separates a music mixture into four stems — **vocals, drums, bass, other** — trained and evaluated on MUSDB18. The pipeline combines energy-based loss weighting, spectrogram-level data augmentation and Wiener-filter post-processing.

The design is based on the configuration studied in:

> D. Kostrzewa, M. Kondziolka, R. Brzeski, J. Abimbola, P. Benecki, *How Far Can a U-Net Go? An Empirical Analysis of Music Source Separation Performance*, Applied Sciences 16(5):2195, 2026. https://doi.org/10.3390/app16052195

Project developed by Aldo Pinelli, Martina Luyten, Hamin Jang and Alessandro Cupini.

---

## Pipeline

### 1. Preprocessing
For every track (100 train / 50 test, official MUSDB18 split):

- stereo → mono (average of the two channels), for the mixture and each of the 4 stems
- resampling from 44.1 kHz to **10,880 Hz** (content up to 5.44 kHz)
- segmentation into **non-overlapping 6 s blocks** (65,280 samples); the last block is taken from the end of the track so the whole song is covered
- **STFT** with a 1022-sample Hann window and hop 256 → magnitude spectrogram of **512 bins × 256 frames**
- the mixture is also resampled along the frequency axis to **256 log-spaced bins** (linear interpolation), giving the **256 × 256** network input

Saved per block: log mixture (network input), linear mixture and linear stem magnitudes (used by the loss). Training set: 3,863 blocks.

### 2. Model
Multi-output U-Net working on the 256 × 256 log-frequency spectrogram:

- **Encoder:** 6 blocks, each with two 3×3 conv + BatchNorm + ReLU, followed by 2×2 max-pooling; channels 32 → 1024
- **Bottleneck:** two 3×3 conv with 2048 channels at 4×4 resolution
- **Decoder:** symmetric, 2×2 transposed convolutions for upsampling, skip connections (concatenation) from the encoder; **Dropout2d (p = 0.5)** in the first 3 decoder blocks
- **Output:** 1×1 conv to 4 channels + sigmoid → one soft mask per source

### 3. Loss
- **Indirect L1 loss in the linear domain:** each predicted mask is interpolated back from 256 × 256 to the original 512 × 256 linear shape, multiplied by the linear mixture magnitude and compared (L1, sum) with the ground-truth stem magnitude.
- **Energy-based weighting (EBW, instantaneous, squared variant):** in every batch each source loss is weighted by

  w<sub>i</sub> = ( max<sub>j</sub> Ē<sub>j</sub> / Ē<sub>i</sub> )²

  where Ē<sub>i</sub> is the mean magnitude of the ground-truth spectrogram of source *i* in the batch. Quieter sources (e.g. bass) get larger weights, so loud sources (e.g. drums) do not dominate the gradient.

### 4. Data augmentation
Applied on the fly in the `Dataset`:

- **Pitch shift (+1 semitone):** every block is added a second time with its linear spectrograms (mixture and stems) shifted up by 30 frequency bins (≈ ratio 2<sup>1/12</sup>) — a simplified, spectrogram-level pitch shift. Doubles the training set to 7,726 blocks.
- **Time–frequency mean masking (p = 0.5):** a random rectangle (up to 60 bins × 40 frames) is replaced by the average of the original region and the same region of the mixture of another random block. The same region of each stem is averaged with that other mixture. The log input is then recomputed.

### 5. Training
| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 1e-3, weight decay (L2) = 0.01 |
| Batch size | 64 |
| Epochs | 1000 (checkpoint every 50 epochs) |
| Hardware | Google Colab GPU, ≈ 2 min/epoch (≈ 33 h total) |

### 6. Reconstruction and post-processing
Each 6 s block of a test track goes through the same STFT/log resampling, the masks are interpolated back to 512 × 256 and applied to the mixture magnitude. The estimates are refined with **one iteration of Wiener filtering** ([norbert](https://github.com/sigsep/norbert)) using the mixture phase, then converted back to audio with the inverse STFT and concatenated.

---

## Evaluation

Metrics: **SDR, SIR, SAR** (BSS Eval) computed with `mir_eval.separation.bss_eval_sources` (BSS Eval v3).

- estimates and references are compared **in mono at 10,880 Hz** (references are downmixed and resampled to match the model output)
- metrics are computed on **1-second windows**; silent reference windows are skipped
- the reported value is the **median over all windows of all 50 test tracks** pooled together


### Results (MUSDB18 test set, 50 tracks, median values in dB)

| Source | SDR | SIR | SAR |
|---|---|---|---|
| Vocals | 5.51 | 8.83 | 9.44 |
| Drums  | 6.29 | 10.73 | 9.01 |
| Bass   | 4.33 | 7.46 | 9.20 |
| Other  | 2.81 | 5.44 | 8.06 |

Drums and vocals, the most prominent sources in a typical mix, are separated best.

---

## Repository content

```
Full_Optimized_MSS.ipynb   # preprocessing, dataset, model, training, evaluation, listening examples
README.md
```

Not included: the MUSDB18 dataset (see its licence), the preprocessed `.npy` files and the trained weights.

## How to run

1. Get MUSDB18 from [sigsep.github.io/datasets/musdb](https://sigsep.github.io/datasets/musdb.html) and put it on Google Drive, e.g. `MyDrive/MUSDB18/musdb18`.
2. Open the notebook in Google Colab with a GPU runtime and run the install cell:
   ```
   pip install musdb museval mir_eval norbert librosa soundfile
   ```
3. Run the cells in order

Paths are hard-coded for Google Drive; change them at the top of each cell if your layout differs.

## Limitations

- Mono, 10.88 kHz processing: the output is band-limited to 5.44 kHz and the stereo image is lost.
- Non-overlapping 6 s blocks with no cross-fade can produce small discontinuities at block borders.
- The spectrogram-level pitch shift is a bin translation, which only approximates a real (multiplicative) pitch shift.
- **Known issue in TF masking:** in the masked region each stem is averaged with the *mixture* of the other block instead of the corresponding *stem*, so the four targets no longer sum to the input mixture there (the other track is over-represented in the targets). The effect is limited — the region covers on average < 1% of the spectrogram and is applied to half of the samples — but it adds some label noise. The reported results were obtained with this version; the fix (averaging each stem with the matching stem of the other block) was not retrained due to compute constraints.

## References

- Kostrzewa et al., *How Far Can a U-Net Go? An Empirical Analysis of Music Source Separation Performance*, Appl. Sci. 2026, 16, 2195.
- Rafii et al., *The MUSDB18 corpus for music separation*, 2017.
- Kadandale et al., *Multi-channel U-Net for Music Source Separation*, MMSP 2020 (energy-based loss weighting).
- Vincent, Gribonval, Févotte, *Performance measurement in blind audio source separation*, IEEE TASLP 2006 (BSS Eval).

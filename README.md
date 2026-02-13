# ComfyUI_MFCC

ComfyUI custom node that computes **MFCC** (Mel-Frequency Cepstral Coefficients) from an audio signal and outputs it as an image.

---

## What are MFCCs?

MFCCs are a compact representation of the short-term power spectrum of audio, widely used in speech recognition, speaker identification, and music analysis.
They capture the **timbral shape** of a signal in a way that closely mirrors human auditory perception.

### Computation Pipeline

```
Raw audio
   │
   ▼
Pre-emphasis  (high-frequency boost: y[t] = x[t] - α·x[t-1])
   │
   ▼
Framing + Hanning window  (overlapping short-time frames)
   │
   ▼
FFT  →  Power spectrum  |X(f)|²
   │
   ▼
Mel filterbank  (triangular filters on the Mel scale)
   │
   ▼
Log energy  (log(mel_power + ε))
   │
   ▼
DCT-II  (decorrelates filter outputs)
   │
   ▼
MFCC coefficients  [n_mfcc × n_frames]
```

---

## Node

### Audio to MFCC

| | |
|---|---|
| **Input** | `AUDIO` (ComfyUI native audio type) |
| **Output** | `IMAGE` — MFCC heatmap `[1, H, W, 3]` float32 |
| **Category** | `audio` |

#### Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `n_mfcc` | 13 | 1 – 128 | Number of MFCC coefficients to compute |
| `n_mels` | 40 | 10 – 256 | Number of Mel filterbank channels |
| `n_fft` | 2048 | 64 – 8192 | FFT size |
| `hop_length` | 512 | 32 – 4096 | Frame shift in samples |
| `win_length` | 2048 | 64 – 8192 | Analysis window length in samples |
| `fmin` | 0.0 Hz | 0 – 4000 | Minimum frequency of the Mel filterbank |
| `fmax` | 8000.0 Hz | 100 – 22050 | Maximum frequency of the Mel filterbank |
| `pre_emphasis` | 0.97 | 0.0 – 1.0 | Pre-emphasis coefficient (0 = disabled) |
| `colormap` | inferno | inferno / magma / viridis / plasma / hot / cool / gray / jet | Matplotlib colormap |
| `width` | 768 | 256 – 4096 | Output image width (px) |
| `height` | 512 | 256 – 4096 | Output image height (px) |
| `show_axes` | ON | ON / OFF | Show time axis, coefficient axis, colorbar, and title |

---

## Output

The output IMAGE tensor has shape `[1, H, W, 3]`, dtype `float32`, values in `[0, 1]`.

- **X-axis** — Time (seconds)
- **Y-axis** — MFCC coefficient index (0 = C0, 1 = C1, …)
- **Color** — Coefficient magnitude mapped through the selected colormap

---

## Typical Settings

| Use case | n_mfcc | n_mels | n_fft | hop_length | fmax |
|---|---|---|---|---|---|
| Speech recognition | 13 | 40 | 2048 | 512 | 8000 |
| Music analysis | 20 | 64 | 2048 | 512 | 11025 |
| High detail / long audio | 40 | 128 | 4096 | 1024 | 16000 |

---

## Installation

### Option A — Clone into custom_nodes directly
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/bemoregt/ComfyUI_MFCC.git
```

### Option B — Symlink from a separate directory
```bash
git clone https://github.com/bemoregt/ComfyUI_MFCC.git
ln -s /path/to/ComfyUI_MFCC /path/to/ComfyUI/custom_nodes/ComfyUI_MFCC
```

Restart ComfyUI after installation.
No additional packages are required beyond those already used by ComfyUI (`numpy`, `matplotlib`, `Pillow`, `torch`).

---

## Example Workflow

```
Load Audio ──► Audio to MFCC ──► Preview Image
```

---

## License

MIT

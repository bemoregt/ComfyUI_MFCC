import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


COLORMAPS = ["inferno", "magma", "viridis", "plasma", "hot", "cool", "gray", "jet"]


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _to_mono(waveform: torch.Tensor) -> np.ndarray:
    """waveform: [1, C, T] → [T] mono float32 numpy"""
    audio = waveform.squeeze(0)
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    return audio.float().numpy()


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                    fmin: float, fmax: float) -> np.ndarray:
    """
    삼각형 Mel 필터뱅크 행렬 반환.
    반환: [n_mels, n_fft//2+1] float64
    """
    # Hz → Mel 변환
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    freq_bins = np.linspace(0, sample_rate / 2, n_bins)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        for k, f in enumerate(freq_bins):
            if f_left <= f <= f_center:
                filterbank[m, k] = (f - f_left) / (f_center - f_left + 1e-10)
            elif f_center < f <= f_right:
                filterbank[m, k] = (f_right - f) / (f_right - f_center + 1e-10)

    return filterbank


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """
    정규화된 DCT-II 행렬 [n_mfcc, n_mels]
    """
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi / n_mels * (n + 0.5) * k)   # [n_mfcc, n_mels]
    dct[0] *= 1.0 / np.sqrt(n_mels)
    dct[1:] *= np.sqrt(2.0 / n_mels)
    return dct


def _compute_mfcc(
    audio_np: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    pre_emphasis: float,
) -> np.ndarray:
    """
    오디오 numpy 배열 → MFCC 행렬
    반환: [n_mfcc, n_frames] float64
    """
    # 1) Pre-emphasis
    if pre_emphasis > 0.0:
        audio_np = np.append(audio_np[0], audio_np[1:] - pre_emphasis * audio_np[:-1])

    # 2) Framing & Windowing
    window = np.hanning(win_length)
    N = len(audio_np)
    frames = []
    for start in range(0, max(N - win_length + 1, 1), hop_length):
        frame = audio_np[start: start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)))
        frames.append(frame * window)
    if not frames:
        frames.append(np.zeros(win_length))
    frames = np.array(frames, dtype=np.float64)   # [n_frames, win_length]

    # 3) FFT → 파워 스펙트럼
    spectra = np.fft.rfft(frames, n=n_fft)        # [n_frames, n_fft//2+1]
    power = np.abs(spectra) ** 2

    # 4) Mel 필터뱅크 적용
    fmax_clip = min(fmax, sample_rate / 2.0)
    filterbank = _mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax_clip)
    mel_power = power @ filterbank.T              # [n_frames, n_mels]

    # 5) Log (에너지)
    log_mel = np.log(mel_power + 1e-10)           # [n_frames, n_mels]

    # 6) DCT → MFCC
    dct = _dct_matrix(n_mfcc, n_mels)            # [n_mfcc, n_mels]
    mfcc = (dct @ log_mel.T)                      # [n_mfcc, n_frames]

    return mfcc


def _render_mfcc(
    mfcc: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    colormap: str,
    width: int,
    height: int,
    show_axes: bool,
) -> torch.Tensor:
    """MFCC 배열 → matplotlib 렌더링 → [1, H, W, 3] float32 tensor"""
    n_mfcc, n_frames = mfcc.shape
    duration = (n_frames * hop_length) / sample_rate

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    im = ax.imshow(
        mfcc,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, n_mfcc],
        cmap=colormap,
        interpolation="nearest",
    )

    if show_axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MFCC Coefficient")
        ax.set_title("MFCC")
        plt.colorbar(im, ax=ax, pad=0.02)
        plt.tight_layout()
    else:
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((width, height), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)   # [1, H, W, 3]


# ──────────────────────────────────────────────
# ComfyUI 노드
# ──────────────────────────────────────────────

class AudioToMFCC:
    """
    ComfyUI 커스텀 노드: 오디오 → MFCC 이미지

    MFCC(Mel-Frequency Cepstral Coefficients) 계산 절차:
      Pre-emphasis → Framing → Hanning window → FFT →
      Mel filterbank → Log → DCT
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_mfcc": (
                    "INT",
                    {
                        "default": 13,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "추출할 MFCC 계수 개수",
                    },
                ),
                "n_mels": (
                    "INT",
                    {
                        "default": 40,
                        "min": 10,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Mel 필터뱅크 채널 수",
                    },
                ),
                "n_fft": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "FFT 크기",
                    },
                ),
                "hop_length": (
                    "INT",
                    {
                        "default": 512,
                        "min": 32,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "프레임 이동 간격 (샘플)",
                    },
                ),
                "win_length": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "윈도우 길이 (샘플), 보통 n_fft와 동일",
                    },
                ),
                "fmin": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4000.0,
                        "step": 50.0,
                        "tooltip": "Mel 필터 최소 주파수 (Hz)",
                    },
                ),
                "fmax": (
                    "FLOAT",
                    {
                        "default": 8000.0,
                        "min": 100.0,
                        "max": 22050.0,
                        "step": 500.0,
                        "tooltip": "Mel 필터 최대 주파수 (Hz)",
                    },
                ),
                "pre_emphasis": (
                    "FLOAT",
                    {
                        "default": 0.97,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "프리엠파시스 계수 (0 = 비활성화)",
                    },
                ),
                "colormap": (COLORMAPS, {"default": "inferno"}),
                "width": (
                    "INT",
                    {"default": 768, "min": 256, "max": 4096, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 256, "max": 4096, "step": 64},
                ),
                "show_axes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mfcc_image",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(
        self,
        audio,
        n_mfcc,
        n_mels,
        n_fft,
        hop_length,
        win_length,
        fmin,
        fmax,
        pre_emphasis,
        colormap,
        width,
        height,
        show_axes,
    ):
        waveform = audio["waveform"]        # [1, C, T]
        sample_rate = audio["sample_rate"]

        audio_np = _to_mono(waveform)

        mfcc = _compute_mfcc(
            audio_np,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            pre_emphasis=pre_emphasis,
        )

        image = _render_mfcc(
            mfcc,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            colormap=colormap,
            width=width,
            height=height,
            show_axes=show_axes,
        )

        return (image,)


# ──────────────────────────────────────────────
# 등록
# ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AudioToMFCC": AudioToMFCC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToMFCC": "Audio to MFCC",
}

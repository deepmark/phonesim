"""Signal-analysis utilities for understanding what a channel does.

The headline function is :func:`analyze_channel`, which compares a clean and a
degraded waveform and returns a dictionary of metrics, including any
caller-supplied ``metrics={name: fn}``. :func:`plot_channel` renders the visual
comparisons to a PNG.

Optional metrics (PESQ, STOI) are used if their packages are installed; absent
ones are reported as ``None`` rather than raising.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch

from phonesim import dsp


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_np_mono(x: ArrayLike) -> np.ndarray:
    """Reduce an audio array to a 1-D mono ``[T]`` signal.

    Accepts ``[T]`` (returned as-is) or ``[C, T]`` (channels averaged). A rank-3
    ``[B, C, T]`` batch is rejected: averaging across the batch axis would merge
    distinct utterances into one meaningless signal, so callers must pass a
    single example (use a per-example loop for batched metrics).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:  # [C, T] -> average channels
        return x.mean(axis=0)
    raise ValueError(
        f"Expected a [T] or [C, T] signal, got rank {x.ndim} {x.shape}. "
        "Pass a single example; compute batched metrics per example."
    )


def _align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Find integer lag (in samples) that best aligns ``b`` to ``a`` via xcorr."""
    a, b = _align_lengths(a, b)
    n = len(a)
    if n == 0:
        return 0
    fa = np.fft.rfft(a, 2 * n)
    fb = np.fft.rfft(b, 2 * n)
    xcorr = np.fft.irfft(fa * np.conj(fb), 2 * n)
    xcorr = np.concatenate([xcorr[-max_lag:], xcorr[: max_lag + 1]])
    lag = int(np.argmax(xcorr)) - max_lag
    return lag


# ----------------------------------------------------------------------------
# Scalar metrics
# ----------------------------------------------------------------------------
def snr_db(clean: np.ndarray, degraded: np.ndarray, align: bool = True) -> float:
    """SNR treating (degraded - clean) as noise, after gain+lag alignment."""
    c, d = _align_lengths(_to_np_mono(clean), _to_np_mono(degraded))
    if align:
        lag = _best_lag(c, d, max_lag=min(len(c) // 2, 2000))
        if lag > 0:
            d = np.concatenate([np.zeros(lag), d])[: len(c)]
        elif lag < 0:
            d = d[-lag:]
            c = c[: len(d)]
        c, d = _align_lengths(c, d)
    # optimal scalar gain to match d to c
    denom = float(np.dot(d, d)) + 1e-12
    g = float(np.dot(c, d)) / denom
    noise = c - g * d
    sig_p = float(np.dot(c, c)) + 1e-12
    noise_p = float(np.dot(noise, noise)) + 1e-12
    return 10.0 * np.log10(sig_p / noise_p)


def band_energy_ratios(x: ArrayLike, sr: int, bands=None) -> dict:
    """Fraction of total energy in each frequency band."""
    if bands is None:
        nyq = sr / 2.0
        # Telephony band edges plus Nyquist; edges above Nyquist are dropped and
        # duplicates merged, so every band has positive width at any sample rate.
        edges = sorted({e for e in (0, 300, 3400, 7000, 14000, nyq) if e <= nyq})
        bands = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
    xv = _to_np_mono(x)
    N = len(xv)
    if N == 0:
        return {}
    spec = np.abs(np.fft.rfft(xv)) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    total = float(spec.sum()) + 1e-12
    out = {}
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"{int(lo)}-{int(hi)}Hz"] = float(spec[mask].sum() / total)
    return out


def highfreq_energy(x: ArrayLike, sr: int, cutoff_hz: float) -> float:
    """Fraction of energy above ``cutoff_hz``. Useful for bandlimit tests."""
    xv = _to_np_mono(x)
    N = len(xv)
    spec = np.abs(np.fft.rfft(xv)) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    total = float(spec.sum()) + 1e-12
    return float(spec[freqs >= cutoff_hz].sum() / total)


def try_pesq(clean: np.ndarray, degraded: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pesq import pesq  # type: ignore
    except Exception:
        return None
    c, d = _align_lengths(_to_np_mono(clean), _to_np_mono(degraded))
    # PESQ supports 8k (narrowband) and 16k (wideband) only.
    target = 16000 if sr >= 16000 else 8000
    mode = "wb" if target == 16000 else "nb"
    ct = dsp.resample(torch.from_numpy(c).view(1, 1, -1).float(), sr, target).view(-1).numpy()
    dt = dsp.resample(torch.from_numpy(d).view(1, 1, -1).float(), sr, target).view(-1).numpy()
    try:
        return float(pesq(target, ct, dt, mode))
    except Exception:
        return None


def try_stoi(clean: np.ndarray, degraded: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pystoi import stoi  # type: ignore
    except Exception:
        return None
    c, d = _align_lengths(_to_np_mono(clean), _to_np_mono(degraded))
    try:
        return float(stoi(c, d, sr, extended=False))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Spectrogram / mel helpers (for plotting)
# ----------------------------------------------------------------------------
def spectrogram_db(x: ArrayLike, sr: int, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    xt = torch.from_numpy(_to_np_mono(x)).float()
    win = torch.hann_window(n_fft)
    spec = torch.stft(xt, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=True)
    mag = spec.abs()
    return 20 * torch.log10(mag + 1e-8).numpy()


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    fmax = sr / 2
    mels = np.linspace(hz_to_mel(0), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for k in range(l, c):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - l) / max(1, (c - l))
        for k in range(c, r):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (r - k) / max(1, (r - c))
    return fb


def logmel(x: ArrayLike, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 64) -> np.ndarray:
    xt = torch.from_numpy(_to_np_mono(x)).float()
    win = torch.hann_window(n_fft)
    spec = torch.stft(xt, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=True)
    power = (spec.abs() ** 2).numpy()
    fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = fb @ power
    return 10 * np.log10(mel + 1e-8)


def frequency_response(clean: ArrayLike, degraded: ArrayLike, sr: int, n_fft: int = 2048):
    """Estimate the magnitude response degraded/clean via averaged periodograms."""
    c = _to_np_mono(clean)
    d = _to_np_mono(degraded)
    c, d = _align_lengths(c, d)
    def avg_spec(x):
        hop = n_fft // 2
        acc = np.zeros(n_fft // 2 + 1)
        cnt = 0
        win = np.hanning(n_fft)
        for start in range(0, len(x) - n_fft, hop):
            seg = x[start : start + n_fft] * win
            acc += np.abs(np.fft.rfft(seg)) ** 2
            cnt += 1
        return acc / max(1, cnt)
    sc = avg_spec(c) + 1e-12
    sd = avg_spec(d) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    resp_db = 10 * np.log10(sd / sc)
    return freqs, resp_db


# ----------------------------------------------------------------------------
# Top-level analysis
# ----------------------------------------------------------------------------
def analyze_channel(
    clean: ArrayLike,
    degraded: ArrayLike,
    sample_rate: int = 24000,
    metrics: Optional[dict[str, Callable]] = None,
    compute_pesq: bool = True,
    compute_stoi: bool = True,
) -> dict:
    """Compute a battery of metrics comparing ``clean`` and ``degraded``.

    Parameters
    ----------
    metrics:
        Optional ``{name: fn}`` of caller-supplied metrics. Each ``fn(audio, sr)``
        returns a float and is evaluated on both signals, reported as
        ``<name>_clean`` and ``<name>_degraded``.

    Returns
    -------
    dict of metric name -> value (``None`` where a metric is unavailable).
    """
    c = _to_np_mono(clean)
    d = _to_np_mono(degraded)
    report: dict = {}
    report["sample_rate"] = sample_rate
    report["len_clean"] = int(len(c))
    report["len_degraded"] = int(len(d))
    report["snr_db"] = snr_db(c, d)
    report["band_energy_clean"] = band_energy_ratios(c, sample_rate)
    report["band_energy_degraded"] = band_energy_ratios(d, sample_rate)
    report["hf_energy_clean_>4k"] = highfreq_energy(c, sample_rate, 4000)
    report["hf_energy_degraded_>4k"] = highfreq_energy(d, sample_rate, 4000)
    report["hf_energy_degraded_>8k"] = highfreq_energy(d, sample_rate, 8000)
    report["pesq"] = try_pesq(c, d, sample_rate) if compute_pesq else None
    report["stoi"] = try_stoi(c, d, sample_rate) if compute_stoi else None

    for name, fn in (metrics or {}).items():
        try:
            report[f"{name}_clean"] = float(fn(clean, sample_rate))
            report[f"{name}_degraded"] = float(fn(degraded, sample_rate))
        except Exception as e:  # pragma: no cover
            report[f"{name}_error"] = str(e)
    return report


def plot_channel(
    clean: ArrayLike,
    degraded: ArrayLike,
    sample_rate: int = 24000,
    path: str = "channel_analysis.png",
    title: str = "Phone-call channel analysis",
):
    """Render waveform, spectrogram, log-mel and frequency-response comparisons."""
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError("plot_channel needs matplotlib (pip install 'phonesim[plot]')") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = _to_np_mono(clean)
    d = _to_np_mono(degraded)
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig.suptitle(title, fontsize=14)

    # waveforms
    tc = np.arange(len(c)) / sample_rate
    td = np.arange(len(d)) / sample_rate
    axes[0, 0].plot(tc, c, lw=0.5); axes[0, 0].set_title("Clean waveform")
    axes[0, 1].plot(td, d, lw=0.5, color="tab:orange"); axes[0, 1].set_title("Degraded waveform")
    for ax in axes[0]:
        ax.set_xlabel("s"); ax.set_ylim(-1.05, 1.05)

    # spectrograms
    sc = spectrogram_db(c, sample_rate)
    sd = spectrogram_db(d, sample_rate)
    vmin, vmax = -80, 0
    for ax, S, ttl in ((axes[1, 0], sc, "Clean spectrogram"), (axes[1, 1], sd, "Degraded spectrogram")):
        im = ax.imshow(S, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[0, S.shape[1], 0, sample_rate / 2], cmap="magma")
        ax.set_title(ttl); ax.set_ylabel("Hz")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # log-mel
    mc = logmel(c, sample_rate)
    md = logmel(d, sample_rate)
    for ax, M, ttl in ((axes[2, 0], mc, "Clean log-mel"), (axes[2, 1], md, "Degraded log-mel")):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(ttl); ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # frequency response + band energy
    freqs, resp = frequency_response(c, d, sample_rate)
    axes[3, 0].plot(freqs, resp); axes[3, 0].set_title("Frequency response (degraded/clean)")
    axes[3, 0].set_xlabel("Hz"); axes[3, 0].set_ylabel("dB"); axes[3, 0].axhline(0, color="k", lw=0.5)
    axes[3, 0].set_ylim(-60, 20)

    be_c = band_energy_ratios(c, sample_rate)
    be_d = band_energy_ratios(d, sample_rate)
    labels = list(be_c.keys())
    xpos = np.arange(len(labels))
    axes[3, 1].bar(xpos - 0.2, [be_c[k] for k in labels], width=0.4, label="clean")
    axes[3, 1].bar(xpos + 0.2, [be_d[k] for k in labels], width=0.4, label="degraded")
    axes[3, 1].set_xticks(xpos); axes[3, 1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[3, 1].set_title("Band energy ratio"); axes[3, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

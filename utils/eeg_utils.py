"""Simple preprocessing helpers for a channel-averaged LFP signal."""

__ALL__ = ['average_channels', 'detrend_signal', 'notch_filter', 'bandpass', 'multitaper_spectrogram', 'multitaper_psd', 'plot_psd', 'plot_spectrogram']
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from __future__ import annotations

from math import gcd
from typing import Optional, Tuple

import numpy as np
from scipy import signal


def average_channels(data_uV: np.ndarray) -> np.ndarray:
    """Average over the channel axis of a ``(n_chan, n_samples)`` array."""
    data_uV = np.asarray(data_uV)
    if data_uV.ndim == 1:
        return data_uV
    if data_uV.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {data_uV.shape}")
    return data_uV.mean(axis=0)


def detrend_signal(x: np.ndarray) -> np.ndarray:
    """Linear detrend along the last axis."""
    return signal.detrend(np.asarray(x), axis=-1, type="linear")


def notch_filter(
    x: np.ndarray,
    fs: float,
    freq: float = 60.0,
    q: float = 30.0,
) -> np.ndarray:
    """IIR notch filter at ``freq`` Hz (default 60 Hz line noise)."""
    b, a = signal.iirnotch(w0=freq / (fs / 2.0), Q=q)
    return signal.filtfilt(b, a, np.asarray(x))


def bandpass(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    if not 0 < low_n < high_n < 1:
        raise ValueError(
            f"Invalid bandpass edges: low={low}, high={high}, fs={fs}"
        )
    sos = signal.butter(order, [low_n, high_n], btype="band", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x))


def _sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """Return a ``(n_windows, win)`` view of ``x`` with given step."""
    x = np.ascontiguousarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {x.shape}")
    if win <= 0 or step <= 0:
        raise ValueError("win and step must be positive")
    if x.size < win:
        raise ValueError(
            f"Signal length {x.size} shorter than window length {win}."
        )
    n_windows = 1 + (x.size - win) // step
    # Explicit copy (not a view) keeps downstream code safe when MNE casts dtypes.
    idx = (np.arange(n_windows)[:, None] * step) + np.arange(win)[None, :]
    return x[idx]

"""Multitaper spectrogram built on ``mne.time_frequency.psd_array_multitaper``.

The signal is cut into overlapping windows, stacked as a 2-D array of
"epochs", and passed once to MNE's multitaper PSD estimator. That gives an
adaptive-weighted, low-bias DPSS spectrogram without a hand-rolled taper
loop.
"""
def multitaper_spectrogram(
    x: np.ndarray,
    fs: float,
    window_s: float = 2.0,
    step_s: float = 0.5,
    bandwidth: float = 4.0,
    fmin: float = 0.5,
    fmax: float = 200.0,
    adaptive: bool = True,
    low_bias: bool = True,
    n_jobs: int = 1,
    return_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a multitaper spectrogram of a 1-D signal.

    Parameters
    ----------
    x
        1-D signal (e.g. the channel-averaged LFP in microvolts).
    fs
        Sample rate in Hz.
    window_s, step_s
        Sliding window length and step in seconds.
    bandwidth
        Frequency smoothing (full bandwidth, Hz). The time-halfbandwidth
        product is ``TW = bandwidth * window_s / 2`` and MNE uses
        ``K = floor(2*TW) - 1`` tapers.
    fmin, fmax
        Frequency range of interest (Hz).
    adaptive
        Use Thomson's adaptive weighting across tapers.
    low_bias
        Drop tapers whose concentration is below 0.9.
    n_jobs
        Parallel jobs (passed to MNE).
    return_db
        If True, return 10*log10(psd + eps); otherwise raw PSD in µV²/Hz.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
    times : ndarray, shape (n_windows,)
        Window-center times in seconds from the start of ``x``.
    S : ndarray, shape (n_freqs, n_windows)
        Spectrogram in dB (or µV²/Hz if ``return_db`` is False).
    """
    # Import here so the rest of the package stays usable if MNE isn't installed.
    from mne.time_frequency import psd_array_multitaper

    x = np.asarray(x, dtype=np.float64)
    win = int(round(window_s * fs))
    step = int(round(step_s * fs))
    epochs = _sliding_windows(x, win, step)  # (n_windows, win)

    psd, freqs = psd_array_multitaper(
        epochs,
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        bandwidth=bandwidth,
        adaptive=adaptive,
        low_bias=low_bias,
        normalization="full",
        n_jobs=n_jobs,
        verbose=False,
    )
    # psd shape: (n_windows, n_freqs) -> transpose to (n_freqs, n_windows)
    S = psd.T

    n_windows = epochs.shape[0]
    t_centers = (np.arange(n_windows) * step + win / 2.0) / fs

    if return_db:
        eps = np.finfo(np.float64).tiny
        S = 10.0 * np.log10(S + eps)

    return freqs, t_centers, S


def multitaper_psd(
    x: np.ndarray,
    fs: float,
    bandwidth: float = 2.0,
    fmin: float = 0.5,
    fmax: float = 200.0,
    adaptive: bool = True,
    low_bias: bool = True,
    n_jobs: int = 1,
    return_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single multitaper PSD of a 1-D signal (no sliding window).

    Parameters
    ----------
    x
        1-D signal.
    fs
        Sample rate in Hz.
    bandwidth
        Frequency smoothing in Hz. With a long signal the default of 2 Hz
        keeps narrow spectral features resolvable; increase for smoother PSDs.
    fmin, fmax
        Frequency range returned.
    adaptive, low_bias, n_jobs
        Passed to MNE.
    return_db
        If True, returns ``10*log10(psd + eps)``; otherwise raw PSD in
        \u00b5V\u00b2/Hz.

    Returns
    -------
    freqs, psd
    """
    from mne.time_frequency import psd_array_multitaper

    x = np.asarray(x, dtype=np.float64)
    psd, freqs = psd_array_multitaper(
        x[np.newaxis, :],
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        bandwidth=bandwidth,
        adaptive=adaptive,
        low_bias=low_bias,
        normalization="full",
        n_jobs=n_jobs,
        verbose=False,
    )
    psd = psd[0]
    if return_db:
        eps = np.finfo(np.float64).tiny
        psd = 10.0 * np.log10(psd + eps)
    return freqs, psd


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    ax=None,
    label: Optional[str] = None,
    log_freq: bool = True,
    in_db: bool = True,
):
    """Plot a single PSD on a (log-)frequency axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, psd, lw=1.2, label=label)
    if log_freq:
        ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB, \u00b5V\u00b2/Hz)" if in_db else "Power (\u00b5V\u00b2/Hz)")
    ax.grid(alpha=0.3, which="both")
    if label is not None:
        ax.legend(loc="best", fontsize=8)
    return ax


def plot_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    S_db: np.ndarray,
    ax=None,
    cmap: str = "magma",
    vlim: Optional[Tuple[float, float]] = None,
    log_freq: bool = False,
    t_offset_s: float = 0.0,
    colorbar: bool = True,
    cbar_label: str = "Power (dB, \u00b5V\u00b2/Hz)",
):
    """Render a spectrogram with ``pcolormesh``.

    Parameters
    ----------
    freqs, times, S_db
        Output of :func:`multitaper_spectrogram`.
    ax
        Matplotlib ``Axes`` to draw into; created if ``None``.
    cmap
        Colormap.
    vlim
        ``(vmin, vmax)`` colour limits. If ``None``, uses the 5th/99th percentiles
        of ``S_db``.
    log_freq
        Draw the frequency axis on a log scale.
    t_offset_s
        Added to ``times`` before plotting (useful if the window didn't start at 0).
    colorbar
        Attach a colour bar to the axis.
    cbar_label
        Label for the colour bar.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if vlim is None:
        vmin, vmax = np.percentile(S_db, [5, 99])
    else:
        vmin, vmax = vlim

    mesh = ax.pcolormesh(
        times + t_offset_s,
        freqs,
        S_db,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if log_freq:
        ax.set_yscale("log")
    if colorbar:
        cb = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label(cbar_label)
    return ax

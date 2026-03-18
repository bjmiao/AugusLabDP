import numpy as np
from scipy.signal import windows, get_window
def multitaper_spectrogram(
    data,
    sfreq,
    win_length=2.0,
    step=0.5,
    time_bandwidth=3.0,
    n_tapers=None,
    fmin=0.5,
    fmax=80.0,
    detrend=True,
    psd_in_db=True,
):
    """
    Compute multitaper spectrogram for EEG.
    Parameters
    ----------
    data : ndarray, shape (n_times,) or (n_channels, n_times)
        EEG data in volts (or µV; units carry through to power).
    sfreq : float
        Sampling frequency in Hz.
    win_length : float
        Window length in seconds.
    step : float
        Step between consecutive windows in seconds.
    time_bandwidth : float
        Time–bandwidth product NW. Typical: 3–4.
    n_tapers : int or None
        Number of DPSS tapers K. If None, uses int(2 * NW - 1).
    fmin, fmax : float
        Min and max frequency (Hz) to keep.
    detrend : bool
        If True, subtract mean from each window before tapering.
    psd_in_db : bool
        If True, return power in dB (10*log10); else linear power.
    Returns
    -------
    times : ndarray, shape (n_windows,)
        Center time of each window in seconds.
    freqs : ndarray, shape (n_freqs,)
        Frequencies in Hz.
    power : ndarray, shape (n_windows, n_freqs) or (n_windows, n_freqs, n_channels)
        Multitaper power spectral density.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[None, :]  # -> (n_channels, n_times)
    n_channels, n_times = data.shape
    win_samples = int(round(win_length * sfreq))
    step_samples = int(round(step * sfreq))
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("win_length and step must be > 0.")
    if n_tapers is None:
        n_tapers = int(2 * time_bandwidth - 1)
        n_tapers = max(n_tapers, 1)
    # DPSS tapers: shape (n_tapers, win_samples)
    tapers = windows.dpss(M=win_samples, NW=time_bandwidth, Kmax=n_tapers, sym=False)
    # Window start indices
    starts = np.arange(0, n_times - win_samples + 1, step_samples, dtype=int)
    n_windows = len(starts)
    if n_windows == 0:
        raise ValueError("Signal too short for given win_length and step.")
    # FFT settings
    n_fft = int(2 ** np.ceil(np.log2(win_samples)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    n_freqs = freqs.size
    power = np.zeros((n_windows, n_freqs, n_channels), dtype=float)
    # Normalization: each taper is L2-normalized by construction;
    # PSD normalization factor accounts for sampling freq and taper count.
    norm_factor = (sfreq * n_tapers)
    for w_i, start in enumerate(starts):
        segment = data[:, start:start + win_samples]  # (n_channels, win_samples)
        if detrend:
            segment = segment - segment.mean(axis=1, keepdims=True)
        # Apply all tapers and accumulate power
        # taper_seg: (n_tapers, n_channels, win_samples)
        taper_seg = tapers[:, None, :] * segment[None, :, :]
        # FFT along time axis
        fft_vals = np.fft.rfft(taper_seg, n=n_fft, axis=-1)  # (n_tapers, n_channels, n_freqs_full)
        fft_vals = fft_vals[..., freq_mask]
        # Power: average |X|^2 across tapers
        seg_power = (np.abs(fft_vals) ** 2).mean(axis=0) / norm_factor  # (n_channels, n_freqs)
        # Store (time, freq, channel)
        power[w_i, :, :] = seg_power.T  # -> (n_freqs, n_channels)
    # Time stamps: centers of each window
    times = (starts + win_samples / 2) / sfreq
    if psd_in_db:
        # Avoid log(0)
        power = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
    # Squeeze channel dimension for single-channel input
    if power.shape[-1] == 1:
        power = power[..., 0]  # (n_windows, n_freqs)
    return times, freqs, power
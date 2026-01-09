import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

__all__ = ['find_r_peaks', 'get_heart_rate', 'plot_ecg_with_r_peaks']

def find_r_peaks(ecg: np.ndarray, sampling_rate: float, max_bpm = 900) -> np.ndarray:
    """
    Find the R peaks in the ECG signal using a Gaussian mixture model.

    """
    peaks = find_peaks(ecg, distance=500)[0]
    # We assume hist is a bimodal distribution, find the gap between two peaks
    # For this, fit a gaussian mixture model to the hist
    gmm = GaussianMixture(n_components=2).fit(ecg[peaks].flatten().reshape(-1, 1))
    means = gmm.means_.flatten()
    search_range = np.arange(means.min(), means.max(), 0.1)
    gmm_values = gmm.predict_proba(search_range.reshape(-1, 1))
    # The gap is where the probability of the two cluster is the same
    gap = np.argmin(np.abs(gmm_values[:, 0] - gmm_values[:, 1]))
    threshold = search_range[gap]
    # Set a BPM hard limit to be 900 BPM
    
    distance = int(sampling_rate / max_bpm * 60)
    r_peaks = find_peaks(ecg, height=threshold, distance=distance)[0]
    r_peaks_in_seconds = r_peaks / sampling_rate
    return r_peaks_in_seconds, threshold

def get_heart_rate(r_peaks_in_seconds: np.ndarray,
                    total_time: float, timebin: float, 
                    temporal_smoothing_window: int = 3) -> np.ndarray:
    """
    Get the heart rate from the R peaks.

    Timebin: in seconds
    """
    bpm = np.histogram(r_peaks_in_seconds, bins=np.arange(0, total_time, timebin))[0].astype(float)
    if temporal_smoothing_window > 0:
        bpm = gaussian_filter(bpm, temporal_smoothing_window)
    return bpm * 60 / timebin

def plot_ecg_with_r_peaks(ecg: np.ndarray, r_peaks_in_seconds: np.ndarray,
                          start_time: float, stop_time: float, sampling_rate: float) -> None:
    """
    Plot the ECG with the R peaks.
    """
    start_index, stop_index = int(start_time * sampling_rate), int(stop_time * sampling_rate)
    fig, ax = plt.subplots(figsize = (10, 3))
    ax.plot(np.arange(start_index, stop_index) / sampling_rate, ecg[start_index:stop_index])
    r_peaks_in_range = r_peaks_in_seconds[(r_peaks_in_seconds > start_time) & (r_peaks_in_seconds < stop_time)]
    ax.scatter(r_peaks_in_range, ecg[(r_peaks_in_range * sampling_rate).astype(int)], color='red', s = 10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ECG (uV)')
    ax.set_title('ECG with R-peaks')
    ax.set_xlim(start_time, stop_time)
    return fig, ax
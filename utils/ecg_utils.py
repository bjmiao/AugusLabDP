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

def get_heart_rate_variability(r_peaks_in_seconds: np.ndarray,
                               total_time: float, timebin: float, 
                               temporal_smoothing_window: int = 3) -> np.ndarray:
    """
    Get the heart rate variability from the R peaks.
    """

    # Calculate RMSSD in each time bin
    bins = np.arange(0, total_time, timebin)
    rmssd = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        # Find all R-peaks within this time bin
        mask = (r_peaks_in_seconds >= bins[i]) & (r_peaks_in_seconds < bins[i + 1])
        r_peaks_in_bin = r_peaks_in_seconds[mask]
        # Calculate RR intervals within the bin
        rr_intervals = np.diff(r_peaks_in_bin)
        # Calculate successive differences
        rr_diff = np.diff(rr_intervals)
        if len(rr_diff) > 0:
            rmssd[i] = np.sqrt(np.mean(rr_diff ** 2))
        else:
            rmssd[i] = np.nan  # Not enough data to compute RMSSD
    # Optional temporal smoothing
    if temporal_smoothing_window > 0:
        rmssd = gaussian_filter(rmssd, temporal_smoothing_window)
    return rmssd

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

def ecg_to_bpm(ecg, sampling_rate):
    r_peaks_in_seconds, threshold = find_r_peaks(ecg, sampling_rate)
    total_time = len(ecg) / sampling_rate
    bpm = get_heart_rate(r_peaks_in_seconds, total_time, timebin = 1, temporal_smoothing_window = 3)
    return bpm
    
if __name__ == '__main__':
    from pathlib import Path
    folder = Path(r'C:\Users\bjmiao\The Augustine Lab Dropbox\Benjie Miao\Benjie_Jonny\SSA_Benjie\DPcachedata\iso\14T_5378529_AP_Amy_Day2_g0')
    ecg_file = folder / 'nidq_ECG.npy'
    ecg = np.load(ecg_file)
    
    sampling_rate = 42372.8
    r_peaks_in_seconds, threshold = find_r_peaks(ecg, sampling_rate)
    # To plot the ECG with the R peaks
    # fig, ax = plot_ecg_with_r_peaks(ecg, r_peaks_in_seconds, start_time = 3000, stop_time = 3020, sample_rate = sampling_rate)
    
    total_time = len(ecg) / sampling_rate
    bpm = get_heart_rate(r_peaks_in_seconds, total_time, timebin = 1, temporal_smoothing_window = 3)
    # To plot the BPM
    # plt.plot(bpm)
    # plt.xlabel('Time (s)')
    # plt.ylabel('BPM')
    # plt.title('BPM over time')
    # plt.legend()
    # plt.show()


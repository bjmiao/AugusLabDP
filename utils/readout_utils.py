"""Utilities for reading and processing neural data."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any

import json
import os

import numpy as np
import pandas as pd


def combine_time_bins(matrix: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """
    Combine time bins in a matrix with shape (#timestep, #neuron).
    
    This function averages consecutive time bins to reduce temporal resolution.
    If the number of timesteps is not divisible by bin_size, the remaining
    timesteps are averaged separately.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix with shape (n_timesteps, n_neurons) or (n_timesteps,).
        If 1D, will be reshaped to 2D internally.
    bin_size : int, default 10
        Number of consecutive time bins to combine.
    
    Returns
    -------
    np.ndarray
        Combined matrix with reduced number of timesteps.
        Shape: (n_timesteps // bin_size + remainder, n_neurons) or 1D if input was 1D.
    """
    need_flatten = False
    if len(matrix.shape) == 1:
        need_flatten = True
        matrix = matrix[:, None]
    num_timesteps, num_neurons = matrix.shape
    num_full_bins = num_timesteps // bin_size
    
    # Reshape and combine full bins
    reshaped = matrix[:num_full_bins*bin_size].reshape(num_full_bins, bin_size, num_neurons)
    combined = np.mean(reshaped, axis=1)
    
    # Handle remaining timesteps if any
    if num_timesteps % bin_size != 0:
        remaining = matrix[num_full_bins*bin_size:]
        remaining_mean = np.mean(remaining, axis=0, keepdims=True)
        combined = np.vstack((combined, remaining_mean))
    
    if need_flatten:
        combined = combined.flatten()
    return combined


def auguslab_manual_correct_ttl_button(
    nidq_ttl_button: np.ndarray,
    nidq_sampling_rate: float,
    session_name: str,
    session_type: str
) -> None:
    """
    Manually correct TTL button signal for specific sessions.
    
    For certain session types (iso_day1, iso_day2), this function corrects
    the TTL button signal by filling in gaps between button press events.
    Modifies the array in-place.
    
    Parameters
    ----------
    nidq_ttl_button : np.ndarray
        TTL button signal array to be corrected (modified in-place).
    nidq_sampling_rate : float
        Sampling rate of the NIDQ system (Hz).
    session_name : str
        Name of the experimental session.
    session_type : str
        Type of session (e.g., 'iso_day1', 'iso_day2', 'test2').
    """
    if session_type == 'iso_day1' or session_type == 'iso_day2':
        # Get the up-edge for the button (TTL laser signal for syncope)
        nidq_ttl_button_up = np.where(
            (nidq_ttl_button[:-1] == 1) & (nidq_ttl_button[1:] == 0)
        )[0]
        # Fill in the gap between first and last button press
        nidq_ttl_button[nidq_ttl_button_up[0]:nidq_ttl_button_up[-1]] = 1

    elif session_name == '53N_5569716_ACC_Amy_Ket_day1_g0' or session_name == '55N_5616735_RSP_AI_Ket_day2_g0':
        first_button_up = np.where(nidq_ttl_button == 1)[0][0]
        time_mask = 5 # in seconds, first button up is failed ketamine IP injection so remove
        nidq_ttl_button[first_button_up:first_button_up + int(time_mask * nidq_sampling_rate)] = 0
        # No correction needed for test2
    else:
        pass

def auguslab_manual_create_experimental_tag(
    results: Dict[str, Any],
    session_name: str,
    session_type: str
) -> Dict[str, Tuple[float, float]]:
    """
    Create experimental period labels based on TTL button signals.
    
    This function analyzes TTL button signals to automatically segment
    experimental sessions into different periods (e.g., Baseline, Laser, Recovery).
    The segmentation logic varies by session type.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing session data including:
        - 'session_start_time': Start time of session (float)
        - 'session_stop_time': Stop time of session (float)
        - 'ttl_button': TTL button signal array, already corrected by the camera timing (np.ndarray)
        - 'ttl_meta': Dictionary with 'niSampRate' key (float)
    session_name : str
        Name of the experimental session.
    session_type : str
        Type of session. Supported types:
        - 'syncope': Syncope protocol with multiple laser stimulations
        - 'ketamine': Ketamine administration protocol
        - 'iso_day1' or 'iso_day2': Isoflurane protocols
        - 'oxy_iso_day1' or 'oxy_iso_day2': Oxygen + Isoflurane protocols
        - '52N_D4': Special protocol for session 52N_D4
    
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping experimental period names to (start_time, stop_time) tuples.
        Times are in seconds relative to session start.
    """
    # session_start_time: float = results['session_start_time']
    # session_stop_time: float = results['session_stop_time']
    session_duration: float = results['session_duration']
    nidq_ttl_button: np.ndarray = results['ttl_button']
    
    # Get the up-edge for the button (TTL laser signal transitions)
    nidq_sampling_rate = float(results['ttl_meta']['niSampRate'])
    nidq_ttl_button_high_time = np.where(nidq_ttl_button == 1)[0] / nidq_sampling_rate
    nidq_ttl_button_up = np.where(
        (nidq_ttl_button[1:] == 1) & (nidq_ttl_button[:-1] == 0)
    )[0]
    nidq_ttl_button_up_time = nidq_ttl_button_up / nidq_sampling_rate
    if session_type == 'syncope':
        # Find 5 big chunks (stimulation trains) separated by gaps > threshold
        threshold = 10  # in seconds, can be any value between 1 and 300
        # Find gaps between button up events
        gaps = nidq_ttl_button_up_time[1:] - nidq_ttl_button_up_time[:-1]
        large_gap_indices = np.where(gaps > threshold)[0]
        
        # Extract start and end times of each stimulation train
        stim_train_start = (
            [nidq_ttl_button_up_time[0]] +
            nidq_ttl_button_up_time[large_gap_indices + 1].tolist()
        )
        stim_train_end = (
            nidq_ttl_button_up_time[large_gap_indices].tolist() +
            [nidq_ttl_button_up_time[-1]]
        )
        stim_train_start = np.array(stim_train_start)
        stim_train_end = np.array(stim_train_end)
        assert len(stim_train_start) == 5, f"Expected 5 stimulation trains, got {len(stim_train_start)}"
        assert len(stim_train_end) == 5, f"Expected 5 stimulation trains, got {len(stim_train_end)}"

        # Define period tags for syncope protocol
        tags = [
            "Baseline", "Laser(Phony)", "Recover 1", "Laser(5Hz)", "Recover 2",
            "Laser(10Hz)", "Recover 3", "Laser(20Hz)", "Recover 4",
            "Laser(20Hz,Long)", "Recover 5"
        ]
        
        # Flatten start/end times and create period boundaries
        all_timelabels = [[start, stop] for start, stop in zip(stim_train_start, stim_train_end)]
        all_timelabels = [x for xs in all_timelabels for x in xs]  # Flatten list
        period_start_timetag = [0] + all_timelabels
        period_end_timetag = all_timelabels + [session_duration]
        
        experimental_label_tag: Dict[str, Tuple[float, float]] = {}
        for tag, period_start, period_end in zip(tags, period_start_timetag, period_end_timetag):
            experimental_label_tag[tag] = (period_start, period_end)
    elif session_type == 'ketamine':
        # Ketamine protocol: single injection event
        ketamine_onset = nidq_ttl_button_up_time[0]
        experimental_label_tag = {
            'Baseline': (0, ketamine_onset),
            'Ketamine': (ketamine_onset, ketamine_onset + 120),
            'Recover': (ketamine_onset + 120, session_duration)
        }
    elif session_type == 'iso_day1' or session_type == 'iso_day2':
        # Isoflurane protocol: single TTL square wave
        iso_onset = nidq_ttl_button_high_time[0]
        iso_offset = nidq_ttl_button_high_time[-1]
        tag = 'Iso'
        if session_type == 'iso_day1':
            tag += '(5%)'
        if session_type == 'iso_day2':
            tag += '(1.5%)'
          
        experimental_label_tag = {
            'Baseline': (0, iso_onset),
            tag: (iso_onset, iso_offset),
            'Recover': (iso_offset, session_duration)
        }
    elif session_type == 'oxy_iso_day1' or session_type == 'oxy_iso_day2':
        # Oxygen + Isoflurane protocol: TTL pulse for oxygen, then isoflurane
        oxy_on_time = nidq_ttl_button_up_time[0]

        # TODO: slightly hardcoded
        threshold = 10 # in seconds, time after the oxygen on
        iso_onset_time = nidq_ttl_button_up_time[np.where(nidq_ttl_button_up_time > oxy_on_time + threshold)[0][0]]
        iso_offset_time = nidq_ttl_button_high_time[-1]
        
        tag = 'Iso'
        if session_type == 'oxy_iso_day1':
            tag += '(5%)'
        if session_type == 'oxy_iso_day2':
            tag += '(1.5%)'
        
        experimental_label_tag = {
            'Baseline': (0, oxy_on_time),
            'Oxygen': (oxy_on_time, iso_onset_time),
            tag: (iso_onset_time, iso_offset_time),
            'Recover': (iso_offset_time, session_duration)
        }
    elif session_type == '52N_D4':
        # Special hardcoded protocol for session 52N_D4
        # TODO: correct for the camera timing
        session_start_time = results['session_start_time']
        experimental_label_tag = {
            'Baseline': (0, 600 - session_start_time),
            'Oxygen': (600 - session_start_time, 2400 - session_start_time),
            'Iso 5%': (2400 - session_start_time, 2460 - session_start_time),
            'Recover1': (2460 - session_start_time, 4200 - session_start_time),
            'Iso 1.5%': (4200 - session_start_time, 6000 - session_start_time),
            'Recover2': (6000 - session_start_time, results['session_duration'])
        }
    else:
        raise ValueError(f"Unknown session_type: {session_type}")
    
    return experimental_label_tag

def auguslab_manual_correct_ttl_camera(
    nidq_ttl_camera: np.ndarray,
    nidq_sampling_rate: float,
    session_name: str,
    session_type: str
) -> None:
    """
    Manually correct TTL camera signal for specific sessions.
    
    For certain sessions, this function corrects the TTL camera signal
    due to signal loss or artifacts. Modifies the array in-place.
    
    Parameters
    ----------
    nidq_ttl_camera : np.ndarray
        TTL camera signal array to be corrected (modified in-place).
    nidq_sampling_rate : float
        Sampling rate of the NIDQ system (Hz).
    session_name : str
        Name of the experimental session.
    session_type : str
        Type of session.
    """
    if session_name == 'test':
        # No correction needed for test session
        pass
    elif session_name == 'test2':
        # No correction needed for test2 session
        pass


def interpolate_array(arr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Interpolate array from its current size to target size using linear interpolation.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to interpolate (1D).
    target_size : int
        Desired size of the output array.
    
    Returns
    -------
    np.ndarray
        Interpolated array of length target_size.
    """
    current_size = len(arr)
    
    # Create indices for the original and target arrays
    old_indices = np.linspace(0, current_size - 1, current_size)
    new_indices = np.linspace(0, current_size - 1, target_size)
    
    # Perform linear interpolation
    interpolated = np.interp(new_indices, old_indices, arr)
    
    return interpolated

def get_cluster_region(
    template_depth: np.ndarray,
    probe_depth_table: pd.DataFrame,
    probe_depth: float = None
) -> List[str]:
    """
    Map cluster template depths to brain regions using probe depth table.
    
    Note: template_depth uses tip (0) to base (max) convention,
    while probe_depth_table uses surface (0) to deep (max) convention.
    The function transforms template_depth to match the probe depth table convention.
    
    Parameters
    ----------
    template_depth : np.ndarray
        Depth for each cluster/unit, measured from tip (0) to base (max).
    probe_depth_table : pd.DataFrame
        Depth table from MATLAB file containing columns:
        - 'start_depth': Start depth of each region (float)
        - 'end_depth': End depth of each region (float)
        - 'acronym': Brain region acronym (str)
    probe_depth: float, optional
        Depth of the probe, used to transform the template depth to the probe depth.
        If None, the maximum depth of the probe depth table is used.
    
    Returns
    -------
    List[str]
        List of brain region acronyms for each cluster.
        Returns 'root' for clusters outside the depth table range.
    """

    
    # Transform template_depth to depth on the probe
    # (convert from tip-to-base to surface-to-deep)
    probe_depth = probe_depth if probe_depth is not None else probe_depth_table['end_depth'].max()
    template_depth_transformed = probe_depth - template_depth

    # Find the region for each unit
    cluster_region: List[str] = []
    for depth in template_depth_transformed:
        # Find all regions where start_depth <= current depth
        matching_regions = np.where(probe_depth_table['start_depth'] <= depth)[0]
        if len(matching_regions) == 0:
            # Not included in the depth table, label as 'root'
            region_id = 'root'
        else:
            # Use the deepest matching region (last index)
            region_id = probe_depth_table['acronym'].iloc[matching_regions[-1]]
        cluster_region.append(region_id)
    return cluster_region

def load_dataset(
    data_folder: str,
    session_name: str,
    session_type: str,
    probe: Union[str, List[str]] = 'all',
    probe_mapping: Dict[str, int] = None,
    need_modules: List[str] = None
) -> Dict[str, Any]:
    """
    Load neural dataset from session folder.
    
    This function loads various data modalities (spikes, video, TTL, EEG, ECG)
    from a session folder and returns them in a structured dictionary.
    
    When TTL camera is presented, the spikes, EEG and ECG and TTL_button needs to be corrected by the video time.
    
    Parameters
    ----------
    data_folder : str
        Path to the data folder containing session subdirectories.
    session_name : str
        Name of the session folder to load.
    session_type : str
        Type of session (e.g., 'syncope', 'ketamine', 'iso_day1').
    probe : Union[str, List[str]], default 'all'
        Which probe(s) to load. 'all' loads all probes found in session folder.
    probe_mapping : Dict[str, int], optional
        Dictionary mapping probe names to probe indices for region assignment.
        If None, defaults to empty dict.
    need_modules : List[str], optional
        List of modules to load. Options: 'spike', 'cluster_region', 'video', 'ttl', 'eeg', 'ecg', 'pupil'
        If None, defaults to all modules.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded data with keys:
        - 'spike_matrix': np.ndarray, shape (n_timepoints, n_neurons)
        - 'ttl_button': np.ndarray, TTL button signal
        - 'ttl_camera': np.ndarray, TTL camera signal
        - 'video_motion': np.ndarray, video motion energy
        - 'video_motSVD': np.ndarray, video motion SVD components
        - 'eeg': np.ndarray, EEG signal
        - 'ecg': np.ndarray, ECG signal
        - 'lfp': np.ndarray, LFP signal
        - 'experimental_label_tag': Dict[str, Tuple[float, float]], experimental periods
        - 'cluster_region': np.ndarray, brain region for each cluster
        - 'has_*': bool flags indicating which modules were successfully loaded
        - 'session_start_time': float, session start time in seconds
        - 'session_stop_time': float, session stop time in seconds
    """
    if probe_mapping is None:
        probe_mapping = {}
    if need_modules is None:
        need_modules = ['spike', 'region', 'video', 'ttl', 'eeg', 'ecg', 'pupil']
    
    session_folder = os.path.join(data_folder, session_name)
    results: Dict[str, Any] = {}
    
    # Load TTL first (needed for session timing)
    if 'ttl' in need_modules:
        try:
            nidq_meta = json.load(open(os.path.join(data_folder, session_name, "nidq_meta.json")))
            results['has_ttl_meta'] = True
            results['ttl_meta'] = nidq_meta
        except FileNotFoundError as e:
            results['has_ttl_meta'] = False
        try:
            nidq_ttl_camera = np.load(os.path.join(data_folder, session_name, "nidq_TTL_Camera.npy"))
            results['has_ttl_camera'] = True
            # Go through a lab-specific manual correction for camera TTL signal due to signal loss in some sessions
            auguslab_manual_correct_ttl_camera(nidq_ttl_camera, float(nidq_meta['niSampRate']), session_name, session_type)
            results['ttl_camera'] = nidq_ttl_camera
            nidq_ttl_camera_high = np.where(nidq_ttl_camera == 1)[0]
            session_start_time = nidq_ttl_camera_high[0] / float(nidq_meta['niSampRate'])
            session_stop_time = nidq_ttl_camera_high[-1] / float(nidq_meta['niSampRate'])
            results['session_start_time'] = session_start_time
            results['session_stop_time'] = session_stop_time
            results['session_duration'] = session_stop_time - session_start_time
        except FileNotFoundError as e:
            results['has_ttl_camera'] = False
            results['session_start_time'] = 0
            results['session_stop_time'] = len(nidq_ttl_camera) / float(nidq_meta['niSampRate'])
            results['session_duration'] = session_stop_time - session_start_time
        try:
            nidq_ttl_button = np.load(os.path.join(data_folder, session_name, "nidq_TTL_Button.npy"))
            results['has_ttl_button'] = True
            auguslab_manual_correct_ttl_button(nidq_ttl_button, float(nidq_meta['niSampRate']), session_name, session_type)
            if results['has_ttl_camera']:
                session_start_index = int(results['session_start_time'] * float(nidq_meta['niSampRate']))
                session_stop_index = int(results['session_stop_time'] * float(nidq_meta['niSampRate']))
                nidq_ttl_button = nidq_ttl_button[session_start_index:session_stop_index]
            results['ttl_button'] = nidq_ttl_button
            experimental_label_tag = auguslab_manual_create_experimental_tag(results, session_name, session_type)
            results['experimental_label_tag'] = experimental_label_tag
        except FileNotFoundError as e:
            results['has_ttl_button'] = False
    else:
        results['has_ttl_meta'] = False
        results['has_ttl_camera'] = False
        results['has_ttl_button'] = False

    # Load spiking matrix first, since it decides the shape of the matrix
    if 'spike' in need_modules:
        all_probes = set()
        spike_timebin = 0.1 # TODO: hardcode for 100msin s
        if probe == 'all':
            for f in os.scandir(session_folder):
                if f.is_dir():
                    all_probes.add(f.path)
            all_spike_matrix = []
            for probe_path in all_probes:
                probe_name = os.path.basename(probe_path)
                spike_matrix = np.load(os.path.join(session_folder, probe_path, "spike_rate_matrix_100ms.npy"))
                all_spike_matrix.append(spike_matrix)
            align_timestep = np.min([matrix.shape[1] for matrix in all_spike_matrix])
            all_spike_matrix = [matrix[:, :align_timestep] for matrix in all_spike_matrix]
            all_spike_matrix = np.concatenate(all_spike_matrix, axis = 0)
            results['spike_matrix'] = all_spike_matrix

            # get the spike time and the clusters for the spikes
<<<<<<< HEAD
            try:
                spike_times_all = []
                for probe_index, probe_path in enumerate(all_probes):
                    probe_name = os.path.basename(probe_path)
                    spike_times = np.load(os.path.join(session_folder, probe_path, "spike_times.npy"))
                    spike_clusters = np.load(os.path.join(session_folder, probe_path, "spike_clusters.npy"))
                    spike_times_all.append((spike_times, spike_clusters))
                results['spike_times'] = spike_times_all
                results['has_spike_times'] = True
            except FileNotFoundError as e:
                results['has_spike_times'] = False

            if results.get('has_cluster_region', True):
                all_cluster_region = np.concatenate(all_cluster_region)
                results['cluster_region'] = all_cluster_region
                results['has_cluster_region'] = True
=======
            spike_times_all = []
            for probe_index, probe_path in enumerate(all_probes):
                probe_name = os.path.basename(probe_path)
                spike_times = np.load(os.path.join(session_folder, probe_path, "spike_times.npy"))
                spike_clusters = np.load(os.path.join(session_folder, probe_path, "spike_clusters.npy"))
                spike_times_all.append((spike_times, spike_clusters))
            results['spike_times'] = spike_times_all
>>>>>>> d0cf752c7f1f3566f5e6e91b6e98085542aa15de
        else:
            raise NotImplementedError
        # If TTL camera is presented, correct the spike matrix by the video time
        if results['has_ttl_camera']:
            session_start_index = int(results['session_start_time'] / spike_timebin)
            session_stop_index = int(results['session_stop_time']  / spike_timebin)
            results['spike_matrix'] = results['spike_matrix'][:, session_start_index:session_stop_index].T
        results['has_spike'] = True
    else:
        results['has_spike'] = False
    
    if 'lfp' in need_modules:
        lfp_fs = 250.0 # TODO: hardcode for 250Hz
        if probe == 'all':
            all_probes = set()
            for f in os.scandir(session_folder):
                if f.is_dir():
                    all_probes.add(f.path)
            all_lfp = []
            for probe_path in all_probes:
                probe_name = os.path.basename(probe_path)
                lfp = np.load(os.path.join(session_folder, probe_path, "lfp_downsample.npy"))
                assert lfp.shape[0] == 385, f"lfp.shape[0] != 385 for {probe_path}"
                lfp = lfp[:384, :] # last channel is SY0 signal, removed
                all_lfp.append(lfp)
            # Find the minimum number of timepoints (columns) across all lfp arrays
            min_timepoints = min(lfp.shape[1] for lfp in all_lfp)
            # Trim each lfp array to have this minimum length along axis 1
            assert all(abs(lfp.shape[1] - min_timepoints) < 5 for lfp in all_lfp), \
                f"One or more lfp arrays are more than 5 samples longer than min_timepoints: {[lfp.shape[1] for lfp in all_lfp]} vs min_timepoints={min_timepoints}"
            all_lfp = [lfp[:, :min_timepoints] for lfp in all_lfp]
            all_lfp = np.concatenate(all_lfp, axis = 0)
            results['lfp'] = all_lfp
            results['has_lfp'] = True
            results['lfp_fs'] = lfp_fs
        else:
            results['has_lfp'] = False

    if 'region' in need_modules:
        if probe == 'all':
            all_probes = set()
            for f in os.scandir(session_folder):
                if f.is_dir():
                    all_probes.add(f.path)
            all_cluster_region = []
            all_channel_region = []
            results['has_cluster_region'] = True
            results['has_channel_region'] = True
            for probe_path in all_probes:
                probe_name = os.path.basename(probe_path)
                # try get the cluster region
                templateDepths = np.load(os.path.join(session_folder, probe_path, "templateDepths.npy"))
                channelDepths = np.repeat(np.arange(0, 3821, 20), 2) # TODO: hardcode for 1.0
       
                probe_index, probe_depth = probe_mapping.get(probe_name, None)
                if probe_index is not None:
                    try:
                        df_depth_table = pd.read_csv(os.path.join(session_folder, f"probe_{probe_index}_location.csv"))
                        results['depth_table'] = df_depth_table
                        cluster_region = get_cluster_region(templateDepths, df_depth_table, probe_depth)
                        channel_region = get_cluster_region(channelDepths, df_depth_table, probe_depth)
                        all_cluster_region.append(cluster_region)
                        all_channel_region.append(channel_region)
                    except Exception as e:
                        print(e)
                        results['has_cluster_region'] = False
            if results.get('has_cluster_region', True) and len(all_cluster_region) > 0:
                all_cluster_region = np.concatenate(all_cluster_region)
                results['cluster_region'] = all_cluster_region
                results['has_cluster_region'] = True
            elif len(all_cluster_region) == 0:
                results['has_cluster_region'] = False
            if results.get('has_channel_region', True) and len(all_channel_region) > 0:
                all_channel_region = np.concatenate(all_channel_region)
                results['channel_region'] = all_channel_region
                results['has_channel_region'] = True
            elif len(all_channel_region) == 0:
                results['has_channel_region'] = False
        else:
            raise NotImplementedError
    if 'video' in need_modules:
        try:
            video_motion = np.load(os.path.join(session_folder, "face_motion.npy"))
            video_motSVD = np.load(os.path.join(session_folder, "face_motSVD.npy"))
            results['has_video'] = True
            spike_matrix = results['spike_matrix']
            motSVD_interpolated_matrix = np.empty((spike_matrix.shape[0], video_motSVD.shape[1]))
            for i in range(video_motSVD.shape[1]):
                motSVD_interpolated_matrix[:, i] = interpolate_array(video_motSVD[:, i], spike_matrix.shape[0])
            video_motion_interpolated = interpolate_array(video_motion, spike_matrix.shape[0])
            results['video_motSVD'] = motSVD_interpolated_matrix
            results['video_motion'] = video_motion_interpolated
        except FileNotFoundError as e:
            results['has_video'] = False
    else:
        results['has_video'] = False
    if 'eeg' in need_modules:
        try:
            nidq_eeg = np.load(os.path.join(session_folder, 'nidq_EEG.npy'))
            if results['has_ttl_camera']:
                nidq_meta = json.load(open(os.path.join(data_folder, session_name, "nidq_meta.json")))
                sampFrequency = float(nidq_meta['niSampRate'])
                session_start_index = int(results['session_start_time'] * sampFrequency)
                session_stop_index = int(results['session_stop_time']  * sampFrequency)
                nidq_eeg = nidq_eeg[session_start_index:session_stop_index]
            results['eeg'] = nidq_eeg
        except FileNotFoundError as e:
            results['has_nidq_eeg'] = False
    else:
        results['has_nidq_eeg'] = False
    if 'ecg' in need_modules:
        try:
            nidq_ecg = np.load(os.path.join(session_folder, 'nidq_ECG.npy'))
            if results['has_ttl_camera']:
                nidq_meta = json.load(open(os.path.join(data_folder, session_name, "nidq_meta.json")))
                sampFrequency = float(nidq_meta['niSampRate'])
                session_start_index = int(results['session_start_time'] * sampFrequency)
                session_stop_index = int(results['session_stop_time']  * sampFrequency)
                nidq_ecg = nidq_ecg[session_start_index:session_stop_index]
            results['ecg'] = nidq_ecg
        except FileNotFoundError as e:
            results['has_nidq_ecg'] = False
    else:
        results['has_nidq_ecg'] = False
    if 'pupil' in need_modules:
        try:
            pupil_data = pd.read_csv(os.path.join(session_folder, 'pupil.csv'), index_col = 0, header = [0, 1, 2])
            results['pupil'] = pupil_data
            results['has_pupil'] = True
        except FileNotFoundError as e:
            results['has_pupil'] = False
    else:
        results['has_pupil'] = False
    return results

# Find the projection point on the line fit by brain_entrance and brain_exit
def find_projection_point(brain_entrance, brain_exit, point):
    distance = np.dot(brain_exit - brain_entrance, point - brain_entrance) / np.dot(brain_exit - brain_entrance, brain_exit - brain_entrance)
    # print(np.sqrt(np.dot(brain_exit - brain_entrance, point - brain_entrance)))
    return brain_entrance + distance * (brain_exit - brain_entrance)

def get_probe_points_info(probe_ccf_file):
    import scipy.io
    probe_ccf = scipy.io.loadmat(probe_ccf_file)

    num_probes = probe_ccf['probe_ccf']['trajectory_coords'].shape[0]
    assert probe_ccf['probe_ccf']['points'].shape[0] == num_probes
    assert probe_ccf['probe_ccf']['trajectory_areas'].shape[0] == num_probes

    probe_points_info = []
    for i in range(num_probes):
        points = probe_ccf['probe_ccf']['points'][i].item()
        trajectory_coords = probe_ccf['probe_ccf']['trajectory_coords'][i].item()
        trajectory_areas = probe_ccf['probe_ccf']['trajectory_areas'][i].item()
        # print(points.shape)
        # print(trajectory_coords.shape)
        # print(trajectory_areas.shape)
        brain_entrance = trajectory_coords[0, :].astype(np.float32)
        brain_exit = trajectory_coords[1, :].astype(np.float32)
        # We find the furtherest point from brain_entrance, to be the depth of the probe

        # Find the line fitted by brain entrance and brain exit, then project the points to the line, and calculate the distance
        # Find the furthest point from the line, to be the depth of the probe
        max_distance = 0
        for point_idx in range(points.shape[0]):

            projection_point = find_projection_point(brain_entrance, brain_exit, points[point_idx, :])
            projection_point = np.array(projection_point, dtype=np.int32)
            distance = np.sqrt(np.dot(projection_point - brain_entrance, projection_point - brain_entrance))

            if distance > max_distance:
                max_distance = distance
        max_distance = max_distance * 10 # brain slice to um
        probe_points_info.append({
            'points': points,
            'trajectory_coords': trajectory_coords,
            'trajectory_areas': trajectory_areas,
            'brain_entrance': brain_entrance,
            'brain_exit': brain_exit,
            'max_distance': max_distance
        })
    return probe_points_info

def get_all_probe_mapping(data_folder: str, datasets: List[str] = ['ketamine','iso', 'syncope']) -> Dict[str, Any]:
    """
    Get the probe mapping for all datasets.
    """
    df_probe_mapping = {}
    for dataset in datasets:
        probe_info_path = os.path.join(data_folder, f"{dataset}_session_mapping.csv")
        df = pd.read_csv(probe_info_path)
        df_probe_mapping[dataset] = df
    return df_probe_mapping


if __name__ == "__main__":
    from pathlib import Path

    # Here is an example of how to use the load_dataset function
    data_folder = Path(r'H:\The Augustine Lab Dropbox\Benjie Miao\Benjie_Jonny\SSA_Benjie\DPcachedata')
    # data_folder = Path(r"C:\Users\bjmiao\The Augustine Lab Dropbox\Benjie Miao\Benjie_Jonny\SSA_Benjie\DPcachedata\\")
    session_info_path = data_folder / 'session_info.csv'
    df_session_info = pd.read_csv(session_info_path)
    print(df_session_info.head())

    df_probe_mapping = get_all_probe_mapping(data_folder)
    [print(dataset, len(df_probe_mapping[dataset])) for dataset in df_probe_mapping.keys()]

    session_index = 10
    item = df_session_info.iloc[session_index]
    session_name = item['session']
    print("Now loading session: ", session_name)
    df = df_probe_mapping[item['dataset']]
    df = df[df.session == item['session']]
    probe_mapping = {probe:(probenum, depth) for probe, probenum, depth in zip(df['probe'], df['probenum'], df['probe_depth'])}
    results = load_dataset(
        data_folder / item['dataset'], item['session'], item['type'],
        probe='all', probe_mapping = probe_mapping,
        need_modules=['cluster_region'])
    print(probe_mapping)
    print(list(results.keys()))
    if 'cluster_region' in results:
        print(results['cluster_region'])
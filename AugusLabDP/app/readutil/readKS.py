"""
Functions for reading Kilosort 4 spike sorting results

Kilosort 4 typically outputs:
- spike_times.npy: spike times in samples
- spike_clusters.npy: cluster IDs for each spike
- amplitudes.npy: spike amplitudes
- templates.npy: spike templates
- cluster_info.tsv: cluster information (optional)
- params.py: sorting parameters
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

def cluster_average(ids, values):
    """
    Replicates MATLAB's clusterAverage:
    For each unique id, compute mean of values belonging to that id.
    Returns an array of size (max_id+1,) with NaN for missing ids.
    """
    ids = np.asarray(ids).astype(int)
    values = np.asarray(values)

    max_id = ids.max()
    out = np.full(max_id + 1, np.nan, dtype=float)

    for uid in np.unique(ids):
        out[uid] = values[ids == uid].mean()

    return out


def template_positions_amplitudes(temps, winv, ycoords,
                                  spikeTemplates, tempScalingAmps):
    """
    Python equivalent of the MATLAB function templatePositionsAmplitudes.
    """

    # temps: (nTemplates, nTime, nChannels)
    # winv: (nChannels, nChannels)
    # ycoords: (nChannels,)
    # spikeTemplates: (nSpikes,)
    # tempScalingAmps: (nSpikes,)

    temps = np.asarray(temps)
    winv = np.asarray(winv)
    ycoords = np.asarray(ycoords).reshape(-1)
    spikeTemplates = np.asarray(spikeTemplates).astype(int)
    tempScalingAmps = np.asarray(tempScalingAmps)

    nTemplates, nTime, nCh = temps.shape

    # Unwhiten templates
    tempsUnW = np.zeros_like(temps)
    for t in range(nTemplates):
        tempsUnW[t] = temps[t] @ winv

    # Channel amplitudes per template
    tempChanAmps = tempsUnW.max(axis=1) - tempsUnW.min(axis=1)   # (nTemplates, nCh)

    # Template amplitude (unscaled)
    tempAmpsUnscaled = tempChanAmps.max(axis=1)

    # Threshold & zero out low channels
    threshVals = tempAmpsUnscaled * 0.3
    tempChanAmps = np.where(tempChanAmps < threshVals[:, None], 0, tempChanAmps)

    # Template depths = center of mass across channels
    templateDepths = (tempChanAmps * ycoords[None, :]).sum(axis=1) / tempChanAmps.sum(axis=1)

    # Spike amplitudes (note MATLAB +1 indexing)
    spikeAmps = tempAmpsUnscaled[spikeTemplates] * tempScalingAmps

    # Compute mean amp per template (true template amplitudes)
    ta = cluster_average(spikeTemplates, spikeAmps)
    tempAmps = ta.copy()   # already indexed by template id

    # Spike depths
    spikeDepths = templateDepths[spikeTemplates]

    # Waveforms: find max-amplitude channel for each template
    max_site = np.argmax(np.max(np.abs(temps), axis=1), axis=1)  # (nTemplates,)
    waveforms = np.zeros((nTemplates, nTime))
    for i in range(nTemplates):
        waveforms[i] = temps[i, :, max_site[i]]

    # Trough-to-peak duration
    trough_idx = np.argmin(waveforms, axis=1)
    templateDuration = np.zeros(nTemplates, dtype=int)
    for i in range(nTemplates):
        seg = waveforms[i, trough_idx[i]:]
        peak_rel = np.argmax(seg)
        templateDuration[i] = peak_rel

    return (spikeAmps,
            spikeDepths,
            templateDepths,
            tempAmps,
            tempsUnW,
            templateDuration,
            waveforms)


def readKS4(ks_folder: Path) -> Dict[str, any]:
    """
    Read Kilosort 4 output files from a folder
    
    Args:
        ks_folder: Path to the Kilosort 4 output folder
        
    Returns:
        Dictionary containing loaded data arrays and metadata
    """
    if not ks_folder.exists() or not ks_folder.is_dir():
        raise ValueError(f"Kilosort folder not found: {ks_folder}")
    
    result = {
        "folder": ks_folder,
    }
        
    result["params"] = _read_params_py(ks_folder / "params.py")
    
    # Read spike_times.npy
    spike_times_path = ks_folder / "spike_times.npy"
    result["spike_times"] = np.load(spike_times_path)
    result["spike_times"] = result["spike_times"] / result["params"]["sample_rate"] # to s

    result["spike_clusters"] = np.load(ks_folder / "spike_clusters.npy")
    result["amplitudes"] = np.load(ks_folder / "amplitudes.npy")
    result["templates"] = np.load(ks_folder / "templates.npy")
    
    winv = np.load(ks_folder / 'whitening_mat_inv.npy')
    coords = np.load(ks_folder / 'channel_positions.npy')
    xcoords, ycoords = coords[:, 0], coords[:, 1]
    spikeTemplates = np.load(ks_folder / 'spike_templates.npy')
    tempScalingAmps = np.load(ks_folder / 'amplitudes.npy')
    spikeAmps, spikeDepths, templateDepths, tempAmps, tempsUnW, templateDuration, waveform = \
                template_positions_amplitudes(result['templates'], winv, ycoords, spikeTemplates, tempScalingAmps)
    
    # Read all the cluster info if available, and combine to a dataframe
    df_cluster_info = None
    for cluster_info_field in ['group', 'Amplitude', 'ContamPct', 'KSLabel']:
        cluster_info_path = ks_folder / f"cluster_{cluster_info_field}.tsv"

        if cluster_info_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(cluster_info_path, sep='\t')
            except ImportError:
                # If pandas is not available, skip cluster_info
                pass
        if df_cluster_info is None:
            df_cluster_info = df
        else:
            df_cluster_info = pd.merge(df_cluster_info, df)
    result['spikeAmps'] = spikeAmps 
    result['spikeDepths'] = spikeDepths
    result['templateDepths'] = templateDepths
    result['tempAmps'] = tempAmps
    result['tempsUnW'] = tempsUnW
    result['templateDuration'] = templateDuration
    result['waveform'] = waveform
    result['df_cluster_info'] = df_cluster_info
        
    return result


def _read_params_py(params_path: Path) -> Dict[str, any]:
    """
    Read Kilosort params.py file and extract parameters
    
    Args:
        params_path: Path to params.py file
        
    Returns:
        Dictionary of parameter values
    """
    params = {}
    try:
        with open(params_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    # Simple parsing of Python assignment statements
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_str = parts[1].strip()
                        # Try to evaluate the value
                        try:
                            value = eval(value_str)
                            params[key] = value
                        except:
                            params[key] = value_str
    except Exception as e:
        print(f"Warning: Could not read params.py: {e}")
    
    return params


def get_num_spikes(ks_data: Dict[str, any]) -> int:
    """Get total number of spikes from Kilosort data"""
    if ks_data.get("spike_times") is not None:
        return len(ks_data["spike_times"])
    return 0


def get_num_clusters(ks_data: Dict[str, any]) -> int:
    """Get number of unique clusters from Kilosort data"""
    if ks_data.get("spike_clusters") is not None:
        return len(np.unique(ks_data["spike_clusters"]))
    return 0

def get_total_time(ks_data: Dict[str, any]) -> float:
    """Get total time of recording from Kilosort data"""
    if ks_data.get("spike_times") is not None:
        return np.max(ks_data["spike_times"])
    return 0

def get_spike_rate_matrix(ks_data: Dict[str, any], bin_size: float) -> np.ndarray:
    """Get spike rate matrix from Kilosort data"""
    num_clusters = get_num_clusters(ks_data)
    total_time = get_total_time(ks_data)
    spike_rate_matrix = np.zeros((num_clusters, int(total_time / bin_size) + 1))
    if ks_data.get("spike_times") is not None:
        for cluster_id in range(num_clusters):
            spikes = get_cluster_spikes(ks_data, cluster_id)
            spike_rate_matrix[cluster_id, :] = np.histogram(spikes, bins=np.arange(0, total_time + bin_size, bin_size))[0] / bin_size
    return spike_rate_matrix

def get_cluster_spikes(ks_data: Dict[str, any], cluster_id: int) -> np.ndarray:
    """
    Get spike times for a specific cluster
    
    Args:
        ks_data: Dictionary from readKS4()
        cluster_id: Cluster ID to extract
        
    Returns:
        Array of spike times for the specified cluster
    """
    if ks_data.get("spike_clusters") is None or ks_data.get("spike_times") is None:
        return np.array([])
    
    mask = ks_data["spike_clusters"] == cluster_id
    return ks_data["spike_times"][mask]


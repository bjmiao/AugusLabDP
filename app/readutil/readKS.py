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
        "spike_times": None,
        "spike_clusters": None,
        "amplitudes": None,
        "templates": None,
        "cluster_info": None,
        "params": None,
    }
    
    # Read spike_times.npy
    spike_times_path = ks_folder / "spike_times.npy"
    if spike_times_path.exists():
        result["spike_times"] = np.load(spike_times_path)
    
    # Read spike_clusters.npy
    spike_clusters_path = ks_folder / "spike_clusters.npy"
    if spike_clusters_path.exists():
        result["spike_clusters"] = np.load(spike_clusters_path)
    
    # Read amplitudes.npy
    amplitudes_path = ks_folder / "amplitudes.npy"
    if amplitudes_path.exists():
        result["amplitudes"] = np.load(amplitudes_path)
    
    # Read templates.npy
    templates_path = ks_folder / "templates.npy"
    if templates_path.exists():
        result["templates"] = np.load(templates_path)
    
    # Read cluster_info.tsv if available
    cluster_info_path = ks_folder / "cluster_info.tsv"
    if cluster_info_path.exists():
        try:
            import pandas as pd
            result["cluster_info"] = pd.read_csv(cluster_info_path, sep='\t')
        except ImportError:
            # If pandas is not available, skip cluster_info
            pass
    
    # Read params.py if available
    params_path = ks_folder / "params.py"
    if params_path.exists():
        result["params"] = _read_params_py(params_path)
    
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


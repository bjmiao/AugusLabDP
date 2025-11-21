"""
Data extractor module for extracting data from various modalities
"""

from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import json

# Import readutil functions
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


try:
    from app.readutil.readKS import readKS4, get_num_spikes, get_num_clusters
    READKS_AVAILABLE = True
except ImportError:
    READKS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

@dataclass
class ExtractionParams:
    """Parameters for data extraction"""
    # Neuropixels AP
    extract_ap: bool = False
    
    # Neuropixels LFP
    extract_lfp: bool = True
    lfp_sampling_freq: float = 250.0  # Hz
    lfp_cutoff_freq: float = 125.0   # Hz
    
    # Spike Sorting
    extract_spikes: bool = True
    spike_rate_bin_size: float = 0.1  # s

    # NIDQ
    extract_nidq: bool = True
    nidq_channels: str = "0,1,2"
    
    # Face Camera
    extract_face: bool = True
    extract_motSVD: bool = True
    extract_movSVD: bool = True
    extract_motion: bool = True

    # Pupil Physiology
    extract_pupil: bool = False
    
    # Probe Location
    extract_probe_location: bool = True


class DataExtractor:
    """Extracts data from various modalities based on parameters"""
    
    def __init__(self, params: ExtractionParams):
        self.params = params
    
    def extract_all(self, current_process_folder: str, sources: list, output_folder: Path) -> Dict[str, Any]:
        """
        Extract all enabled data sources

        Args:
            current_process_folder: Name of the current process folder
                will be the parent folder of the extracted data
                and the name of the folder in the output folder
            sources: List of DataSource objects to extract
            output_folder: Path to save extracted data
            
        Returns:
            Dictionary with extraction results
        """
        results = {}
        # Store the parameters in the root output folder since it is universal
        with open(output_folder / 'params.json', 'w') as f:
            json.dump(self.params.__dict__, f)
        output_folder = output_folder / current_process_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        # Store a json file, 'params.json' in the output folder, with the parameters

        for source in sources:
            if not source.enabled:
                continue
            
            try:
                if source.data_type == "Neuropixels AP" and self.params.extract_ap:
                    results[source.name] = self._extract_ap(source, output_folder)
                elif source.data_type == "Neuropixels LFP" and self.params.extract_lfp:
                    results[source.name] = self._extract_lfp(source, output_folder)
                elif source.data_type == "Spike Sorting" and self.params.extract_spikes:
                    results[source.name] = self._extract_spikes(source, output_folder)
                elif source.data_type == "NIDQ" and self.params.extract_nidq:
                    results[source.name] = self._extract_nidq(source, output_folder)
                elif source.data_type == "Face Camera" and self.params.extract_face:
                    results[source.name] = self._extract_face(source, output_folder)
                elif source.data_type == "Pupil Physiology" and self.params.extract_pupil:
                    results[source.name] = self._extract_pupil(source, output_folder)
                elif source.data_type == "Probe Location" and self.params.extract_probe_location:
                    results[source.name] = self._extract_probe_location(source, output_folder)
            except Exception as e:
                results[source.name] = {"error": str(e)}
        
        return results
    
    def _extract_ap(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract Neuropixels AP data"""
        # TODO: Implement AP extraction
        print(f"Extracting AP data from {source.path}")
        return {"status": "not_implemented", "message": "AP extraction not yet implemented"}
    
    def _extract_lfp(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract Neuropixels LFP data"""
        # TODO: Implement LFP extraction with filtering
        print(f"Extracting LFP data from {source.path}")
        return {
            "status": "not_implemented",
            "message": f"LFP extraction with sampling_freq={self.params.lfp_sampling_freq} Hz, "
                      f"cutoff_freq={self.params.lfp_cutoff_freq} Hz not yet implemented"
        }
    
    def _extract_spikes(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract spike sorting data"""
        try:
            from app.readutil.readKS import readKS4, \
                get_num_spikes, get_num_clusters, get_total_time, get_spike_rate_matrix
        except ImportError:
            return {"status": "error", "message": "readKS module not found"}
        # TODO: At this point we output the time binned spike rate. Later also store templates and the spike trains
        ks_data = readKS4(source.path)
        num_spikes = get_num_spikes(ks_data)
        num_clusters = get_num_clusters(ks_data)
        total_time = get_total_time(ks_data)
        spike_rate_matrix = get_spike_rate_matrix(ks_data, self.params.spike_rate_bin_size)
        np.save(output_folder / f"spike_rate_matrix_{int(self.params.spike_rate_bin_size * 1000)}ms.npy", spike_rate_matrix)
        return {"status": "success", "message": f"Spike data extracted for {num_spikes} spikes in {total_time} seconds"}
    
    def _extract_nidq(self, source, output_folder: Path) -> Dict[str, Any]:# Import readutil functions
        try:
            from app.readutil.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractAnalog, ExtractDigital
        except ImportError:
            return {"error": "readSGLX module not found"}

        """Extract NIDQ data"""
        # Parse channel list
        try:
            channels = [int(ch.strip()) for ch in self.params.nidq_channels.split(',')]
            print(f"Extracting NIDQ data from {source.path} for channels {channels}")
        except ValueError:
            return {"error": f"Invalid channel format: {self.params.nidq_channels}"}
        # TODO: hard coded here for the channel to extraction mapping
        meta = readMeta(source.bin_file)
        # Get number of channels and file samples
        nChan = int(meta['nSavedChans'])
        fileSizeBytes = int(meta['fileSizeBytes'])
        nFileSamp = int(fileSizeBytes / (2 * nChan))  # 2 bytes per int16 sample
        
        rawData = makeMemMapRaw(source.bin_file, meta)
        firstSamp, lastSamp = 0, nFileSamp - 1 # both end inclusive
        status = "success"
        failed_channels = []
        for channel in channels:
            try:
                if channel == 0: # ECG signal
                    convArray = ExtractAnalog(rawData, [channel], firstSamp, lastSamp, meta)
                    convArray = convArray.flatten()
                    np.save(output_folder / f"nidq_ECG.npy", convArray)
                elif channel == 1: # EEG signal
                    convArray = ExtractAnalog(rawData, [channel], firstSamp, lastSamp, meta)
                    convArray = convArray.flatten()
                    np.save(output_folder / f"nidq_EEG.npy", convArray)
                elif channel == 2: # TTL Camera signal
                    digArray = ExtractDigital(rawData, firstSamp, lastSamp, 0, [1], meta)[0]
                    digArray = digArray.flatten()
                    np.save(output_folder / f"nidq_TTL_Camera.npy", digArray)
            except Exception as e:
                print(e)
                failed_channels.append(channel)
                status = "error"
        if status == "error":
            return {"status": "error", "message": f"Failed to extract NIDQ data for channels {failed_channels}"}
        else:
            return {"status": "success", "message": f"NIDQ data extracted for channels {channels}"}
    
    def _extract_face(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract face camera data"""
        print(source.path)
        face_data = np.load(str(source.path), allow_pickle=True).item()
        if self.params.extract_motSVD:
            motSVD = face_data['motSVD'][1]
            np.save(output_folder / f"face_motSVD.npy", motSVD)
        if self.params.extract_movSVD:
            movSVD = face_data['movSVD'][1]
            np.save(output_folder / f"face_movSVD.npy", movSVD)
        if self.params.extract_motion:
            motion = face_data['motion'][1]
            np.save(output_folder / f"face_motion.npy", motion)
        return {"status": "success", "message": f"Face data extracted for fields facemap"}
    
    def _extract_pupil(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract pupil physiology data"""
        # TODO: Implement pupil extraction
        print(f"Extracting pupil data from {source.path}")
        return {"status": "not_implemented", "message": "Pupil extraction not yet implemented"}

    def _extract_probe_location(self, source, output_folder: Path) -> Dict[str, Any]:
        try:
            from scipy.io import loadmat
        except ImportError:
            return {"status": "error", "message": "scipy is not installed"}

        try:
            mat_data = loadmat(str(source.path))
        except Exception as e:
            return {"status": "error", "message": f"Error loading depth table file: {e}"}

        depth_table = None
        for key in mat_data.keys():
            if not key.startswith('__'):  # Skip MATLAB metadata keys
                depth_table = mat_data[key]
                break
        # Convert to numpy array if needed
        if not isinstance(depth_table, np.ndarray):
            depth_table = np.array(depth_table)
        depth_table = depth_table.flatten()

        # Convert the depth table to a dictionary of probe information
        all_probes_dict = []
        for probe_id in range(depth_table.shape[0] - 1): # last field is path
            probe_info_dict = {
                'start_depth': [],
                'end_depth': [],
                'acronym': [],
                'full_name': [],
                'region_id': []
            }
            for segment_id in range(depth_table[probe_id].shape[0]):
                start_depth = depth_table[probe_id][segment_id][0].item()
                end_depth = depth_table[probe_id][segment_id][1].item()
                acronym = depth_table[probe_id][segment_id][2].item()[0]
                full_name = depth_table[probe_id][segment_id][3].item()[0]
                region_id = depth_table[probe_id][segment_id][4].item()
                probe_info_dict['start_depth'].append(start_depth)
                probe_info_dict['end_depth'].append(end_depth)
                probe_info_dict['acronym'].append(acronym)
                probe_info_dict['full_name'].append(full_name)
                probe_info_dict['region_id'].append(region_id)
            all_probes_dict.append(probe_info_dict)
        np.save(output_folder / f"all_probes_location.npy", all_probes_dict, allow_pickle=True)
        return {"status": "success", "message": f"Probe location data extracted for {len(all_probes_dict)} probes"}
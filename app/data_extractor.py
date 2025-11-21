"""
Data extractor module for extracting data from various modalities
"""

from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np


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
    
    # NIDQ
    extract_nidq: bool = True
    nidq_channels: str = "0,1,2"
    
    # Face Camera
    extract_face: bool = True
    extract_motSVD: bool = True
    extract_movSVD: bool = True
    
    # Pupil Physiology
    extract_pupil: bool = True


class DataExtractor:
    """Extracts data from various modalities based on parameters"""
    
    def __init__(self, params: ExtractionParams):
        self.params = params
    
    def extract_all(self, sources: list, output_folder: Path) -> Dict[str, Any]:
        """
        Extract all enabled data sources
        
        Args:
            sources: List of DataSource objects to extract
            output_folder: Path to save extracted data
            
        Returns:
            Dictionary with extraction results
        """
        results = {}
        output_folder.mkdir(parents=True, exist_ok=True)
        
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
            except Exception as e:
                results[source.name] = {"error": str(e)}
        
        return results
    
    def _extract_ap(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract Neuropixels AP data"""
        # TODO: Implement AP extraction
        return {"status": "not_implemented", "message": "AP extraction not yet implemented"}
    
    def _extract_lfp(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract Neuropixels LFP data"""
        # TODO: Implement LFP extraction with filtering
        return {
            "status": "not_implemented",
            "message": f"LFP extraction with sampling_freq={self.params.lfp_sampling_freq} Hz, "
                      f"cutoff_freq={self.params.lfp_cutoff_freq} Hz not yet implemented"
        }
    
    def _extract_spikes(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract spike sorting data"""
        # TODO: Implement spike extraction
        return {"status": "not_implemented", "message": "Spike extraction not yet implemented"}
    
    def _extract_nidq(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract NIDQ data"""
        # Parse channel list
        try:
            channels = [int(ch.strip()) for ch in self.params.nidq_channels.split(',')]
        except ValueError:
            return {"error": f"Invalid channel format: {self.params.nidq_channels}"}
        
        # TODO: Implement NIDQ extraction
        return {
            "status": "not_implemented",
            "message": f"NIDQ extraction for channels {channels} not yet implemented"
        }
    
    def _extract_face(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract face camera data"""
        # TODO: Implement face extraction
        extract_fields = []
        if self.params.extract_motSVD:
            extract_fields.append("motSVD")
        if self.params.extract_movSVD:
            extract_fields.append("movSVD")
        
        return {
            "status": "not_implemented",
            "message": f"Face extraction for fields {extract_fields} not yet implemented"
        }
    
    def _extract_pupil(self, source, output_folder: Path) -> Dict[str, Any]:
        """Extract pupil physiology data"""
        # TODO: Implement pupil extraction
        return {"status": "not_implemented", "message": "Pupil extraction not yet implemented"}


"""
Data detector module for scanning and detecting available data sources
in Neuropixels preprocessing folders
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataSource:
    """Represents a detected data source"""
    name: str
    data_type: str
    path: Path
    enabled: bool = True
    description: str = ""
    bin_file: Optional[Path] = None  # Path to the actual bin file if applicable
    
    def __str__(self):
        return f"{self.name} ({self.data_type})"


class DataDetector:
    """Detects available data sources in a Neuropixels data folder"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.sources: List[DataSource] = []
    
    def scan(self) -> List[DataSource]:
        """Scan the folder and detect all available data sources"""
        self.sources = []
        
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            return self.sources
        
        # Scan for imec folders (Neuropixels AP/LFP data)
        self._scan_imec_folders()
        
        # Scan for NIDQ files (TTL and EEG/ECG data)
        self._scan_nidq_files()
        
        # Scan for face behavioral data (.npy files starting with 'face_')
        self._scan_face_data()
        
        # Scan for pupil data (.csv files starting with 'pupil_')
        self._scan_pupil_data()
        
        # Scan for depth table
        self._scan_depth_table()
        
        return self.sources
    
    def _scan_imec_folders(self):
        """Scan for imec folders containing AP/LFP data"""
        # Pattern to match folders ending with imec followed by a number
        imec_pattern = re.compile(r'imec\d+$')

        for item in self.folder_path.iterdir():
            if item.is_dir() and imec_pattern.search(item.name):
                # Find AP files using pattern matching
                ap_bin_pattern = re.compile(r'.*\.ap\.bin$')
                ap_meta_pattern = re.compile(r'.*\.ap\.meta$')

                ap_bin_file = None
                ap_meta_file = None
                for file_item in item.iterdir():
                    if file_item.is_file():
                        if ap_bin_pattern.match(file_item.name):
                            ap_bin_file = file_item
                        elif ap_meta_pattern.match(file_item.name):
                            ap_meta_file = file_item
                
                if ap_bin_file and ap_meta_file:
                    self.sources.append(DataSource(
                        name=f"{item.name} - AP Data",
                        data_type="Neuropixels AP",
                        path=item,
                        description=f"Action potential data from {item.name}",
                        bin_file=ap_bin_file
                    ))
                
                # Find LFP files using pattern matching
                lfp_bin_pattern = re.compile(r'.*\.lf\.bin$')
                lfp_meta_pattern = re.compile(r'.*\.lf\.meta$')

                lfp_bin_file = None
                lfp_meta_file = None
                for file_item in item.iterdir():
                    if file_item.is_file():
                        if lfp_bin_pattern.match(file_item.name):
                            lfp_bin_file = file_item
                        elif lfp_meta_pattern.match(file_item.name):
                            lfp_meta_file = file_item
                
                if lfp_bin_file and lfp_meta_file:
                    self.sources.append(DataSource(
                        name=f"{item.name} - LFP Data",
                        data_type="Neuropixels LFP",
                        path=item,
                        description=f"Local field potential data from {item.name}",
                        bin_file=lfp_bin_file
                    ))
                
                # Find Kilosort folders using pattern matching
                kilosort_pattern = re.compile(r'kilosort\d*$', re.IGNORECASE)        
                for dir_item in item.iterdir():
                    if dir_item.is_dir() and kilosort_pattern.match(dir_item.name):
                        kilosort_name = dir_item.name
                        self.sources.append(DataSource(
                            name=f"{item.name} - {kilosort_name.capitalize()} Results",
                            data_type="Spike Sorting",
                            path=dir_item,
                            description=f"{kilosort_name.capitalize()} spike sorting results from {item.name}"
                        ))
    
    def _scan_nidq_files(self):
        """Scan for NIDQ files (TTL and EEG/ECG data)"""
        for item in self.folder_path.iterdir():
            if item.is_file():
                # Look for .nidq.bin and .nidq.meta files
                if item.suffix == '.bin' and '.nidq' in item.name:
                    meta_file = self.folder_path / f"{item.stem}.meta"
                    if meta_file.exists():
                        self.sources.append(DataSource(
                            name=f"NIDQ Data ({item.name})",
                            data_type="NIDQ",
                            path=item.parent,
                            description="TTL signals and EEG/ECG physiology data",
                            bin_file=item
                        ))
                        break  # Only add once per NIDQ dataset
    
    def _scan_face_data(self):
        """Scan for face behavioral data (.npy files include 'face_' case-insensitive, including subfolders)"""
        # Search recursively in the folder and subfolders
        for item in self.folder_path.rglob('*.npy'):
            if item.is_file():
                # Case-insensitive check for files starting with 'face_'
                name_lower = item.name.lower()
                if 'face' in name_lower:
                    self.sources.append(DataSource(
                        name=f"Face Behavioral Data ({item.name})",
                        data_type="Face Camera",
                        path=item,
                        description="Facial camera behavioral metrics"
                    ))
    
    def _scan_pupil_data(self):
        """Scan for pupil data (.csv files include 'pupil_' case-insensitive, including subfolders)"""
        # Search recursively in the folder and subfolders
        for item in self.folder_path.rglob('*.csv'):
            if item.is_file():
                # Case-insensitive check for files starting with 'pupil_'
                name_lower = item.name.lower()
                if 'pupil' in name_lower:
                    self.sources.append(DataSource(
                        name=f"Pupil Data ({item.name})",
                        data_type="Pupil Physiology",
                        path=item,
                        description="Pupil size measurements over time"
                    ))
    
    def _scan_depth_table(self):
        """Scan for depth table file"""
        depth_table = self.folder_path / "depth_table.mat"
        if depth_table.exists():
            self.sources.append(DataSource(
                name="Depth Table (depth_table.mat)",
                data_type="Probe Location",
                path=depth_table,
                description="Probe insertion location data"
            ))
    
    def get_enabled_sources(self) -> List[DataSource]:
        """Get only the enabled data sources"""
        return [source for source in self.sources if source.enabled]


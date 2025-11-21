"""
Data analyzer module for providing quick overviews of data sources
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import readutil functions
try:
    from app.readutil.readSGLX import readMeta, SampRate
    READSGLX_AVAILABLE = True
except ImportError:
    READSGLX_AVAILABLE = False

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


class DataAnalyzer:
    """Analyzes data sources and provides quick overviews"""
    
    @staticmethod
    def get_overview(data_type: str, path: Path, bin_file: Optional[Path] = None) -> Dict[str, any]:
        """
        Get a quick overview of a data source
        
        Args:
            data_type: Type of data source
            path: Path to the data source
            bin_file: Optional path to the bin file (for AP/LFP/NIDQ data)
            
        Returns:
            Dictionary with overview information
        """
        if data_type == "Probe Location":
            return DataAnalyzer._analyze_depth_table(path)
        elif data_type in ["Neuropixels AP", "Neuropixels LFP", "NIDQ"]:
            if bin_file is None:
                return {"error": "Bin file path not provided"}
            return DataAnalyzer._analyze_bin_file(bin_file, data_type)
        elif data_type == "Spike Sorting":
            return DataAnalyzer._analyze_kilosort(path)
        elif data_type == "Face Camera":
            return DataAnalyzer._analyze_face_camera(path)
        elif data_type == "Pupil Physiology":
            return DataAnalyzer._analyze_pupil_data(path)
        # Add more analyzers for other data types as needed
        return {"status": "Overview not yet implemented"}
    
    @staticmethod
    def _analyze_depth_table(path: Path) -> Dict[str, any]:
        """Analyze depth_table.mat file"""
        if not SCIPY_AVAILABLE:
            return {
                "error": "scipy is required to read .mat files. Please install it: pip install scipy"
            }
        
        if not path.exists():
            return {"error": "File not found"}
        
        try:
            # Load the .mat file
            mat_data = loadmat(str(path))
            
            # Find the depth table data
            # Usually stored as 'depth_table' or similar key
            depth_table = None
            for key in mat_data.keys():
                if not key.startswith('__'):  # Skip MATLAB metadata keys
                    depth_table = mat_data[key]
                    break
            
            if depth_table is None:
                return {"error": "Could not find depth table data in .mat file"}
            
            # Convert to numpy array if needed
            if not isinstance(depth_table, np.ndarray):
                depth_table = np.array(depth_table)
            
            # Get shape information
            if depth_table.ndim == 2:
                num_rows, num_columns = depth_table.shape
                num_probes = num_columns - 1
                
                return {
                    "columns": num_columns,
                    "rows": num_rows,
                    "probes": num_probes,
                    "shape": depth_table.shape,
                    "status": "success"
                }
            else:
                return {
                    "error": f"Unexpected array shape: {depth_table.shape}. Expected 2D array."
                }
                
        except Exception as e:
            return {
                "error": f"Error reading depth table: {str(e)}"
            }
    
    @staticmethod
    def _analyze_bin_file(bin_path: Path, data_type: str) -> Dict[str, any]:
        """Analyze SpikeGLX bin file"""
        if not READSGLX_AVAILABLE:
            return {
                "error": "readSGLX module not available"
            }
        
        if not bin_path.exists():
            return {"error": "Bin file not found"}
        
        try:
            # Read metadata
            meta = readMeta(bin_path)
            
            if not meta:
                return {"error": "Could not read metadata file"}
            
            # Get number of channels and file samples
            nChan = int(meta['nSavedChans'])
            fileSizeBytes = int(meta['fileSizeBytes'])
            nFileSamp = int(fileSizeBytes / (2 * nChan))  # 2 bytes per int16 sample
            
            # Get sampling rate
            sRate = SampRate(meta)
            
            # Calculate total recording time in seconds
            total_time_sec = nFileSamp / sRate
            
            # Format time nicely
            if total_time_sec < 60:
                time_str = f"{total_time_sec:.1f}s"
            elif total_time_sec < 3600:
                time_str = f"{total_time_sec/60:.1f}min"
            else:
                hours = int(total_time_sec // 3600)
                minutes = int((total_time_sec % 3600) // 60)
                time_str = f"{hours}h {minutes}min"
            
            return {
                "nChan": nChan,
                "nFileSamp": nFileSamp,
                "sampling_rate": sRate,
                "total_time_sec": total_time_sec,
                "time_str": time_str,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Error reading bin file: {str(e)}"
            }
    
    @staticmethod
    def _analyze_kilosort(ks_path: Path) -> Dict[str, any]:
        """Analyze Kilosort 4 output folder"""
        if not READKS_AVAILABLE:
            return {
                "error": "readKS module not available"
            }
        
        if not ks_path.exists() or not ks_path.is_dir():
            return {"error": "Kilosort folder not found"}
        
        try:
            # Read Kilosort data
            ks_data = readKS4(ks_path)
            
            num_spikes = get_num_spikes(ks_data)
            num_clusters = get_num_clusters(ks_data)
            
            return {
                "num_spikes": num_spikes,
                "num_clusters": num_clusters,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Error reading Kilosort data: {str(e)}"
            }
    
    @staticmethod
    def _format_time(total_time_sec: float) -> str:
        """Format time in seconds to a readable string"""
        if total_time_sec < 60:
            return f"{total_time_sec:.1f}s"
        elif total_time_sec < 3600:
            return f"{total_time_sec/60:.1f}min"
        else:
            hours = int(total_time_sec // 3600)
            minutes = int((total_time_sec % 3600) // 60)
            return f"{hours}h {minutes}min"
    
    @staticmethod
    def _analyze_face_camera(face_path: Path) -> Dict[str, any]:
        """Analyze face camera behavioral data (.npy file)"""
        if not face_path.exists():
            return {"error": "Face camera file not found"}
        
        try:
            # Load the npy file with allow_pickle=True
            face_data = np.load(str(face_path), allow_pickle=True).item()
            # Try to extract motSVD, movSVD, and motion fields
            motSVD = face_data['motSVD'][1]
            movSVD = face_data['movSVD'][1]
            motion = face_data['motion'][1]
            # Get frame count from motSVD if available
            num_frames = None
            if motSVD is not None:
                if isinstance(motSVD, np.ndarray):
                    num_frames = motSVD.shape[0]
                elif hasattr(motSVD, 'shape'):
                    num_frames = motSVD.shape[0]
            elif movSVD is not None:
                if isinstance(movSVD, np.ndarray):
                    num_frames = movSVD.shape[0]
                elif hasattr(movSVD, 'shape'):
                    num_frames = movSVD.shape[0]
            elif motion is not None:
                if isinstance(motion, np.ndarray):
                    num_frames = motion.shape[0]
                elif hasattr(motion, 'shape'):
                    num_frames = motion.shape[0]
            else:
                # Fallback: try to get shape from the main array
                if isinstance(face_data, np.ndarray) and face_data.ndim > 0:
                    num_frames = face_data.shape[0]
            
            if num_frames is None:
                return {"error": "Could not determine number of frames from face camera data"}
            
            # Calculate recording time at 30 Hz
            frame_rate = 30.0  # Default frame rate
            total_time_sec = num_frames / frame_rate
            time_str = DataAnalyzer._format_time(total_time_sec)
            
            return {
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "total_time_sec": total_time_sec,
                "time_str": time_str,
                "status": "success"
            }
            
        except Exception as e:
            print(e)
            return {
                "error": f"Error reading face camera data: {str(e)}"
            }
    
    @staticmethod
    def _analyze_pupil_data(pupil_path: Path) -> Dict[str, any]:
        """Analyze pupil physiology data (.csv file)"""
        if not PANDAS_AVAILABLE:
            return {
                "error": "pandas is required to read CSV files. Please install it: pip install pandas"
            }
        
        if not pupil_path.exists():
            return {"error": "Pupil data file not found"}
        
        try:
            # Load the CSV file using pandas
            df = pd.read_csv(pupil_path, index_col = 0, header = [0, 1, 2])
            
            # Count number of rows (frames)
            num_frames = len(df)
            
            # Calculate recording time at 30 Hz
            frame_rate = 30.0  # Default frame rate
            total_time_sec = num_frames / frame_rate
            time_str = DataAnalyzer._format_time(total_time_sec)
            
            return {
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "total_time_sec": total_time_sec,
                "time_str": time_str,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Error reading pupil data: {str(e)}"
            }


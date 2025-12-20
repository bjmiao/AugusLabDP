"""
Widget for displaying and configuring processing options
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QScrollArea,
    QFrame,
    QGroupBox,
)
from PyQt6.QtCore import Qt
from app.data_extractor import ExtractionParams


class ModalityOptionsWidget(QWidget):
    """Widget for configuring extraction options for a single modality"""
    
    def __init__(self, modality_name: str, parent=None):
        super().__init__(parent)
        self.modality_name = modality_name
        self.params = {}
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Main checkbox for enabling this modality
        self.enable_checkbox = QCheckBox(f"Extract {modality_name}")
        layout.addWidget(self.enable_checkbox)
        
        # Container for parameter fields
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_layout.setContentsMargins(20, 5, 10, 5)
        self.params_container.setLayout(self.params_layout)
        layout.addWidget(self.params_container)
        
        self.setLayout(layout)
    
    def add_float_field(self, label: str, default: float, unit: str = ""):
        """Add a float input field"""
        field_layout = QHBoxLayout()
        field_label = QLabel(f"{label}:")
        field_label.setMinimumWidth(120)
        field_layout.addWidget(field_label)
        
        field_input = QLineEdit(str(default))
        field_input.setValidator(None)  # We'll validate on extraction
        field_layout.addWidget(field_input)
        
        if unit:
            unit_label = QLabel(unit)
            field_layout.addWidget(unit_label)
        
        field_layout.addStretch()
        self.params_layout.addLayout(field_layout)
        
        self.params[label] = field_input
        return field_input
    
    def add_string_field(self, label: str, default: str):
        """Add a string input field"""
        field_layout = QHBoxLayout()
        field_label = QLabel(f"{label}:")
        field_label.setMinimumWidth(120)
        field_layout.addWidget(field_label)
        
        field_input = QLineEdit(default)
        field_layout.addWidget(field_input)
        field_layout.addStretch()
        self.params_layout.addLayout(field_layout)
        
        self.params[label] = field_input
        return field_input
    
    def add_checkbox_field(self, label: str, default: bool = True):
        """Add a checkbox field"""
        checkbox = QCheckBox(label)
        checkbox.setChecked(default)
        self.params_layout.addWidget(checkbox)
        
        self.params[label] = checkbox
        return checkbox
    
    def is_enabled(self) -> bool:
        """Check if this modality is enabled for extraction"""
        return self.enable_checkbox.isChecked()
    
    def get_params(self) -> dict:
        """Get current parameter values"""
        result = {}
        for key, widget in self.params.items():
            if isinstance(widget, QLineEdit):
                result[key] = widget.text()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
        return result


class ProcessingOptionsWidget(QWidget):
    """Widget for displaying all processing options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modality_widgets: dict[str, ModalityOptionsWidget] = {}
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Processing Options")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Scroll area for options
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container widget
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_layout.setSpacing(10)
        self.container.setLayout(self.container_layout)
        
        scroll_area.setWidget(self.container)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
        
        # Initialize with default modalities
        self._initialize_modalities()
    
    def _initialize_modalities(self):
        """Initialize all modality options with default settings"""
        # Neuropixels AP
        ap_widget = ModalityOptionsWidget("Neuropixels AP")
        self.modality_widgets["Neuropixels AP"] = ap_widget
        self.container_layout.addWidget(ap_widget)
        
        # Neuropixels LFP
        lfp_widget = ModalityOptionsWidget("Neuropixels LFP")
        lfp_widget.add_float_field("Sampling Frequency", 250.0, "Hz")
        lfp_widget.add_float_field("Cutoff frequency", 125.0, "Hz")
        self.modality_widgets["Neuropixels LFP"] = lfp_widget
        self.container_layout.addWidget(lfp_widget)
        
        # Spike Sorting
        spikes_widget = ModalityOptionsWidget("Spike Sorting")
        spikes_widget.add_float_field("Spike rate bin size", 0.1, "s")
        self.modality_widgets["Spike Sorting"] = spikes_widget
        self.container_layout.addWidget(spikes_widget)
        
        # NIDQ
        nidq_widget = ModalityOptionsWidget("NIDQ")
        nidq_widget.add_string_field("Channels", "0,1,2,3")
        self.modality_widgets["NIDQ"] = nidq_widget
        self.container_layout.addWidget(nidq_widget)
        
        # Face Camera
        face_widget = ModalityOptionsWidget("Face Camera")
        face_widget.add_checkbox_field("motSVD", default=True)
        face_widget.add_checkbox_field("movSVD", default=True)
        face_widget.add_checkbox_field("motion", default=True)
        self.modality_widgets["Face Camera"] = face_widget
        self.container_layout.addWidget(face_widget)
        
        # Pupil Physiology
        pupil_widget = ModalityOptionsWidget("Pupil Physiology")
        self.modality_widgets["Pupil Physiology"] = pupil_widget
        self.container_layout.addWidget(pupil_widget)
        
        # Probe Location (Depth Table)
        probe_location_widget = ModalityOptionsWidget("Probe Location")
        self.modality_widgets["Probe Location"] = probe_location_widget
        self.container_layout.addWidget(probe_location_widget)

        # Set default Checked state by the ExtractionParam
        params = ExtractionParams()
        ap_widget.enable_checkbox.setChecked(params.extract_ap)
        lfp_widget.enable_checkbox.setChecked(params.extract_lfp)
        spikes_widget.enable_checkbox.setChecked(params.extract_spikes)
        nidq_widget.enable_checkbox.setChecked(params.extract_nidq)
        face_widget.enable_checkbox.setChecked(params.extract_face)
        # face_widget["motSVD"].setChecked(params.extract_motSVD) # TODO how to access subwidget
        # face_widget["movSVD"].setChecked(params.extract_movSVD)
        # face_widget["motion"].setChecked(params.extract_motion)
        pupil_widget.enable_checkbox.setChecked(params.extract_pupil)
        probe_location_widget.enable_checkbox.setChecked(params.extract_probe_location)

        # Add stretch at the end
        self.container_layout.addStretch()
    
    def get_extraction_params(self) -> ExtractionParams:
        """Get extraction parameters from all modality widgets"""
        params = ExtractionParams()
        
        # Neuropixels AP
        if "Neuropixels AP" in self.modality_widgets:
            params.extract_ap = self.modality_widgets["Neuropixels AP"].is_enabled()
        
        # Neuropixels LFP
        if "Neuropixels LFP" in self.modality_widgets:
            lfp_widget = self.modality_widgets["Neuropixels LFP"]
            params.extract_lfp = lfp_widget.is_enabled()
            lfp_params = lfp_widget.get_params()
            try:
                params.lfp_sampling_freq = float(lfp_params.get("Sampling Frequency", 250.0))
                params.lfp_cutoff_freq = float(lfp_params.get("Cutoff frequency", 125.0))
            except ValueError:
                pass  # Keep defaults if invalid
        
        # Spike Sorting
        if "Spike Sorting" in self.modality_widgets:
            params.extract_spikes = self.modality_widgets["Spike Sorting"].is_enabled()
            params.spike_rate_bin_size = float(self.modality_widgets["Spike Sorting"].get_params().get("Spike rate bin size", 0.01))
        # NIDQ
        if "NIDQ" in self.modality_widgets:
            nidq_widget = self.modality_widgets["NIDQ"]
            params.extract_nidq = nidq_widget.is_enabled()
            nidq_params = nidq_widget.get_params()
            params.nidq_channels = nidq_params.get("Channels", "0,1,2,3")
        
        # Face Camera
        if "Face Camera" in self.modality_widgets:
            face_widget = self.modality_widgets["Face Camera"]
            params.extract_face = face_widget.is_enabled()
            face_params = face_widget.get_params()
            params.extract_motSVD = face_params.get("motSVD", True)
            params.extract_movSVD = face_params.get("movSVD", True)
            params.extract_motion = face_params.get("motion", True)
        
        # Pupil Physiology
        if "Pupil Physiology" in self.modality_widgets:
            params.extract_pupil = self.modality_widgets["Pupil Physiology"].is_enabled()
        
        # Probe Location
        if "Probe Location" in self.modality_widgets:
            params.extract_probe_location = self.modality_widgets["Probe Location"].is_enabled()
        
        return params


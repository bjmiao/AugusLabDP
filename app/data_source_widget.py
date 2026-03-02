"""
Widget for displaying and selecting data sources
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
)
from app.data_detector import DataSource
from app.data_analyzer import DataAnalyzer


class DataSourceItem(QWidget):
    """Widget for a single data source item"""
    
    def __init__(self, source: DataSource, parent=None):
        super().__init__(parent)
        self.source = source
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 5)
        
        # Vertical layout for name, type, and overview
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        # Name and type in horizontal layout
        name_type_layout = QHBoxLayout()
        name_type_layout.setContentsMargins(0, 0, 0, 0)
        
        # Name label
        name_label = QLabel(source.name)
        name_label.setStyleSheet("font-weight: bold;")
        name_type_layout.addWidget(name_label)
        
        # Type label
        type_label = QLabel(f"({source.data_type})")
        type_label.setStyleSheet("color: gray;")
        name_type_layout.addWidget(type_label)
        
        # Label badge (if available)
        if hasattr(source, 'label') and source.label:
            label_badge = QLabel(f"[{source.label}]")
            label_badge.setStyleSheet("color: #0066cc; font-weight: bold; font-size: 10px; padding: 2px 6px; background-color: #e6f2ff; border-radius: 3px;")
            name_type_layout.addWidget(label_badge)
        
        name_type_layout.addStretch()
        info_layout.addLayout(name_type_layout)
        
        # Overview label (will be populated with analysis results)
        self.overview_label = QLabel()
        self.overview_label.setStyleSheet("color: #0066cc; font-size: 11px; font-style: italic;")
        info_layout.addWidget(self.overview_label)
        
        main_layout.addLayout(info_layout)
        
        # Spacer
        main_layout.addStretch()
        
        # Path label (truncated if too long)
        path_text = str(source.path)
        if len(path_text) > 60:
            path_text = "..." + path_text[-57:]
        path_label = QLabel(path_text)
        path_label.setStyleSheet("color: #666; font-size: 10px;")
        path_label.setToolTip(str(source.path))
        main_layout.addWidget(path_label)
        
        self.setLayout(main_layout)
        
        # Add border
        self.setStyleSheet("""
            DataSourceItem {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            DataSourceItem:hover {
                background-color: #f5f5f5;
            }
        """)
        
        # Load overview information
        self._load_overview()
    
    def _load_overview(self):
        """Load and display overview information for this data source"""
        # Pass bin_file if available
        bin_file = getattr(self.source, 'bin_file', None)
        overview = DataAnalyzer.get_overview(self.source.data_type, self.source.path, bin_file)
        
        if "error" in overview:
            self.overview_label.setText(f"⚠ {overview['error']}")
            self.overview_label.setStyleSheet("color: #cc0000; font-size: 11px;")
        elif "status" in overview and overview["status"] == "success":
            # Format overview based on data type
            if self.source.data_type == "Probe Location":
                probes = overview.get("probes", "?")
                columns = overview.get("columns", "?")
                self.overview_label.setText(f"📊 {probes} probes ({columns} columns)")
            elif self.source.data_type in ["Neuropixels AP", "Neuropixels LFP", "NIDQ"]:
                nChan = overview.get("nChan", "?")
                time_str = overview.get("time_str", "?")
                sRate = overview.get("sampling_rate", 0)
                self.overview_label.setText(
                    f"📊 {nChan} channels | {sRate:.1f} Hz | Duration {time_str}"
                )
            elif self.source.data_type == "Spike Sorting":
                num_spikes = overview.get("num_spikes", "?")
                num_clusters = overview.get("num_clusters", "?")
                self.overview_label.setText(
                    f"📊 {num_spikes:,} spikes | {num_clusters} clusters"
                )
            elif self.source.data_type == "Face Camera":
                num_frames = overview.get("num_frames", "?")
                time_str = overview.get("time_str", "?")
                self.overview_label.setText(
                    f"📊 {num_frames:,} frames | Duration {time_str}"
                )
            elif self.source.data_type == "Pupil Physiology":
                num_frames = overview.get("num_frames", "?")
                time_str = overview.get("time_str", "?")
                self.overview_label.setText(
                    f"📊 {num_frames:,} frames | Duration {time_str}"
                )
            else:
                self.overview_label.setText("")
        else:
            self.overview_label.setText("")
    
class DataSourceListWidget(QWidget):
    """Widget for displaying a list of data sources"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_items: list[DataSourceItem] = []
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Detected Data Sources")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Scroll area for the list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container widget for source items
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_layout.setSpacing(5)
        self.container_layout.addStretch()
        self.container.setLayout(self.container_layout)
        
        scroll_area.setWidget(self.container)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
    
    def set_sources(self, sources: list[DataSource]):
        """Set the data sources to display"""
        # Clear existing items
        for item in self.source_items:
            self.container_layout.removeWidget(item)
            item.deleteLater()
        self.source_items.clear()
        
        # Add new items
        for source in sources:
            item = DataSourceItem(source)
            self.source_items.append(item)
            # Insert before the stretch
            self.container_layout.insertWidget(
                self.container_layout.count() - 1,
                item
            )
    
    def get_enabled_sources(self) -> list[DataSource]:
        """Get all data sources (all are enabled by default)"""
        return [item.source for item in self.source_items]

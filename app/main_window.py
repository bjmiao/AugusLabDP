"""
Main window for the AugusLab Data Preprocessing Dashboard
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSplitter,
)
from PyQt6.QtCore import Qt
from pathlib import Path
from app.data_detector import DataDetector
from app.data_source_widget import DataSourceListWidget


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AugusLab Data Preprocessing Dashboard")
        self.setGeometry(100, 100, 1400, 900)
        
        self.current_folder: Path = None
        self.detector: DataDetector = None
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Header section
        header_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("AugusLab Data Preprocessing Dashboard")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        header_layout.addWidget(title_label)
        
        # Folder selection section
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(10)
        
        # Load folder button
        self.load_button = QPushButton("Select Data Folder")
        self.load_button.setMinimumHeight(40)
        self.load_button.setMinimumWidth(150)
        self.load_button.clicked.connect(self.load_data_folder)
        folder_layout.addWidget(self.load_button)
        
        # Current folder label
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: gray; padding: 5px;")
        folder_layout.addWidget(self.folder_label)
        
        folder_layout.addStretch()
        header_layout.addLayout(folder_layout)
        
        main_layout.addLayout(header_layout)
        
        # Splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Data source list
        self.data_source_widget = DataSourceListWidget()
        splitter.addWidget(self.data_source_widget)
        
        # Right side: Processing options
        from app.processing_options_widget import ProcessingOptionsWidget
        self.processing_options_widget = ProcessingOptionsWidget()
        splitter.addWidget(self.processing_options_widget)
        
        # Set splitter proportions (60% left, 40% right)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select a data folder to begin")
    
    def load_data_folder(self):
        """Open folder dialog to select data folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            self.current_folder = Path(folder_path)
            self.folder_label.setText(f"Folder: {self.current_folder.name}")
            self.folder_label.setToolTip(str(self.current_folder))
            self.folder_label.setStyleSheet("color: black; padding: 5px;")
            
            # Scan for data sources
            self.statusBar().showMessage("Scanning folder for data sources...")
            self.detector = DataDetector(str(self.current_folder))
            sources = self.detector.scan()
            
            if sources:
                self.data_source_widget.set_sources(sources)
                self.statusBar().showMessage(
                    f"Found {len(sources)} data source(s) - Select which ones to process"
                )
            else:
                self.data_source_widget.set_sources([])
                QMessageBox.warning(
                    self,
                    "No Data Sources Found",
                    "No recognized data sources were found in the selected folder.\n\n"
                    "Please ensure the folder contains:\n"
                    "- imec folders with .ap.bin/.lfp.bin files\n"
                    "- .nidq.bin files\n"
                    "- face_*.npy files\n"
                    "- pupil_*.csv files\n"
                    "- depth_table.mat"
                )
                self.statusBar().showMessage("No data sources found")


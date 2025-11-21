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
    QListWidget,
    QProgressBar,
    QLineEdit,
)
from PyQt6.QtCore import Qt
from pathlib import Path
from app.data_detector import DataDetector
from app.data_extractor import DataExtractor
from app.data_source_widget import DataSourceListWidget


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AugusLab Data Preprocessing Dashboard")
        self.setGeometry(100, 100, 1600, 700)
        
        self.folders: list[Path] = []
        self.current_folder: Path = None
        self.detector: DataDetector = None
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Header section (shorter)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("AugusLab Data Preprocessing Dashboard")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Main content area with splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Folder list
        folder_panel = QWidget()
        folder_panel_layout = QVBoxLayout()
        folder_panel_layout.setContentsMargins(5, 5, 5, 5)
        
        folder_list_label = QLabel("Dataset Folders")
        folder_list_label.setStyleSheet("font-weight: bold;")
        folder_panel_layout.addWidget(folder_list_label)
        
        # Folder list box
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.folder_list.itemSelectionChanged.connect(self._on_folder_selected)
        folder_panel_layout.addWidget(self.folder_list)
        
        # Folder management buttons
        folder_buttons_layout = QHBoxLayout()
        self.add_folders_button = QPushButton("Add Folder(s)")
        self.add_folders_button.clicked.connect(self.add_folders)
        folder_buttons_layout.addWidget(self.add_folders_button)
        
        self.remove_folders_button = QPushButton("Remove Folder(s)")
        self.remove_folders_button.clicked.connect(self.remove_folders)
        folder_buttons_layout.addWidget(self.remove_folders_button)
        
        folder_panel_layout.addLayout(folder_buttons_layout)
        folder_panel.setLayout(folder_panel_layout)
        folder_panel.setMaximumWidth(250)
        main_splitter.addWidget(folder_panel)
        
        # Middle and right: Data sources and processing options
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Middle: Data source list
        self.data_source_widget = DataSourceListWidget()
        content_splitter.addWidget(self.data_source_widget)
        
        # Right side: Processing options
        from app.processing_options_widget import ProcessingOptionsWidget
        self.processing_options_widget = ProcessingOptionsWidget()
        content_splitter.addWidget(self.processing_options_widget)
        
        # Set content splitter proportions (60% left, 40% right)
        content_splitter.setSizes([600, 400])
        main_splitter.addWidget(content_splitter)
        
        # Set main splitter proportions (15% left, 85% right)
        main_splitter.setSizes([200, 1200])
        main_layout.addWidget(main_splitter)
        
        # Bottom: Output folder, Extract button and progress bar
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(5, 5, 5, 5)
        
        # Output folder selection
        output_folder_layout = QHBoxLayout()
        output_folder_label = QLabel("Output Folder:")
        output_folder_label.setMinimumWidth(100)
        output_folder_layout.addWidget(output_folder_label)
        
        self.output_folder_line = QLineEdit()
        # Set default output folder to repository root / cachedata
        repo_root = Path(__file__).parent.parent  # Go up from app/main_window.py to project root
        default_output = repo_root / "cachedata"
        self.output_folder_line.setText(str(default_output))
        output_folder_layout.addWidget(self.output_folder_line)
        
        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.setMaximumWidth(80)
        self.browse_output_button.clicked.connect(self.browse_output_folder)
        output_folder_layout.addWidget(self.browse_output_button)
        
        bottom_layout.addLayout(output_folder_layout)
        
        # Extract button
        self.extract_button = QPushButton("Start Extracting Data")
        self.extract_button.setMinimumHeight(35)
        self.extract_button.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.extract_button.clicked.connect(self.start_extraction)
        bottom_layout.addWidget(self.extract_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        bottom_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(bottom_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready - Add folders to begin")
    
    def browse_output_folder(self):
        """Open folder dialog to select output folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            self.output_folder_line.setText(folder_path)
            self.statusBar().showMessage(f"Output folder set: {folder_path}")
    
    def add_folders(self):
        """Open folder dialog to add one or more folders"""
        # Note: QFileDialog.getExistingDirectory only supports single selection
        # Users can call this multiple times to add multiple folders
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder (You can add multiple folders by clicking 'Add Folder(s)' again)",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            folder_path = Path(folder_path)
            
            # Check if folder is already in the list
            if folder_path not in self.folders:
                self.folders.append(folder_path)
                self.folder_list.addItem(str(folder_path))
                self.statusBar().showMessage(f"Added folder: {folder_path.name}")
                
                # Auto-select the newly added folder
                items = self.folder_list.findItems(str(folder_path), Qt.MatchFlag.MatchExactly)
                if items:
                    self.folder_list.setCurrentItem(items[0])
            else:
                QMessageBox.information(
                    self,
                    "Folder Already Added",
                    f"Folder '{folder_path.name}' is already in the list."
                )
    
    def remove_folders(self):
        """Remove selected folders from the list"""
        selected_items = self.folder_list.selectedItems()
        
        if not selected_items:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select folder(s) to remove."
            )
            return
        
        # Remove from list (in reverse order to maintain indices)
        for item in reversed(selected_items):
            folder_path = Path(item.text())
            if folder_path in self.folders:
                self.folders.remove(folder_path)
            row = self.folder_list.row(item)
            self.folder_list.takeItem(row)
        
        # If current folder was removed, clear data sources
        if self.current_folder and self.current_folder not in self.folders:
            self.current_folder = None
            self.data_source_widget.set_sources([])
            self.statusBar().showMessage("Folder removed - Select another folder to view data")
        
        self.statusBar().showMessage(f"Removed {len(selected_items)} folder(s)")
    
    def _on_folder_selected(self):
        """Handle folder selection from list"""
        selected_items = self.folder_list.selectedItems()
        
        if not selected_items:
            return
        
        # Use the first selected item
        selected_item = selected_items[0]
        folder_path = Path(selected_item.text())
        
        if folder_path == self.current_folder:
            return  # Already loaded
        
        self.current_folder = folder_path
        
        # Scan for data sources
        self.statusBar().showMessage(f"Scanning folder: {folder_path.name}...")
        self.detector = DataDetector(str(self.current_folder))
        sources = self.detector.scan()
        
        if sources:
            self.data_source_widget.set_sources(sources)
            self.statusBar().showMessage(
                f"Found {len(sources)} data source(s) in {folder_path.name}"
            )
        else:
            self.data_source_widget.set_sources([])
            QMessageBox.warning(
                self,
                "No Data Sources Found",
                f"No recognized data sources were found in '{folder_path.name}'.\n\n"
                "Please ensure the folder contains:\n"
                "- imec folders with .ap.bin/.lfp.bin files\n"
                "- .nidq.bin files\n"
                "- face_*.npy files\n"
                "- pupil_*.csv files\n"
                "- depth_table.mat"
            )
            self.statusBar().showMessage(f"No data sources found in {folder_path.name}")
    
    def start_extraction(self):
        """Start data extraction process"""
        if not self.folders:
            QMessageBox.warning(
                self,
                "No Folders",
                "Please add at least one folder before starting extraction."
            )
            return
        
        # Get and validate output folder
        output_folder_text = self.output_folder_line.text().strip()
        if not output_folder_text:
            QMessageBox.warning(
                self,
                "No Output Folder",
                "Please select an output folder for the extracted data."
            )
            return
        current_process_folder = self.current_folder.absolute().name
        print(current_process_folder)
        output_folder = Path(output_folder_text)
        
        # Get extraction parameters
        params = self.processing_options_widget.get_extraction_params()
        data_extractor = DataExtractor(params)

        # Get enabled sources from all folders
        all_sources = []
        for folder in self.folders:
            detector = DataDetector(str(folder))
            sources = detector.scan()
            all_sources.extend([s for s in sources if s.enabled])
        
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Extraction started... Output: {output_folder}")
        if not all_sources:
            QMessageBox.warning(
                self,
                "No Sources Selected",
                "Please enable at least one data source in the selected folders."
            )
            return
        
        results = data_extractor.extract_all(current_process_folder, all_sources, output_folder)
        # Show the results in a message box, showing "#success / #total" in the log
        success_count = 0
        for result in results.values():
            if result.get("status", "Undefined") == "success":
                success_count += 1
        print(results)

        self.statusBar().showMessage(f"{success_count} / {len(results)} sources extracted successfully")

"""
AugusLab Data Preprocessing Dashboard
Main entry point for the application
"""

import sys
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("AugusLab Data Preprocessing Dashboard")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


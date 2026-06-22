"""Console entry point for the AugusLabDP dashboard."""

import sys

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("AugusLab Data Preprocessing Dashboard")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

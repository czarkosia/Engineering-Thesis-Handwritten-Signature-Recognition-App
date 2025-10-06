import sys
from PyQt6.QtWidgets import QApplication

from ui import MainWindow

STYLES_FILE = "ui/styles/styles.qss"

if __name__ == "__main__":
    app = QApplication(sys.argv)

    with open(STYLES_FILE, 'r') as f:
        app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
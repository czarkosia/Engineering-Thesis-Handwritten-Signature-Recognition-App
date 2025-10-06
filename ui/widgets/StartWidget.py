from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout


class StartWidget(QWidget):
    goToRegister = pyqtSignal()
    goToVerify = pyqtSignal()
    goToIdentify = pyqtSignal()

    def __init__(self, parent=None):
        super(StartWidget, self).__init__(parent)
        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.goToRegister.emit)

        self.verify_button = QPushButton("Verify")
        self.verify_button.clicked.connect(self.goToVerify.emit)
        self.identify_button = QPushButton("Identify")
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.register_button)
        self.layout().addWidget(self.verify_button)
        self.layout().addWidget(self.identify_button)

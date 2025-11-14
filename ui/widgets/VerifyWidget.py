from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel

from ui.widgets.SignatureCanvasWidget import SignatureCanvasWidget


class VerifyWidget(QWidget):
    goToStart = pyqtSignal()

    def __init__(self, parent=None):
        super(VerifyWidget, self).__init__(parent)
        self.signatureCanvas = SignatureCanvasWidget()

        self.nameLabel = QLabel("Name:")
        self.nameBox = QLineEdit(self)

        self.verifyButton = QPushButton("Verify")
        self.verifyButton.clicked.connect(self.verify)

        self.abortButton = QPushButton("Abort")
        self.abortButton.clicked.connect(self.goToStart)

        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.signatureCanvas.clear)

        menuLayout = QHBoxLayout()
        menuLayout.addWidget(self.nameLabel)
        menuLayout.addWidget(self.nameBox)
        menuLayout.addWidget(self.verifyButton)
        menuLayout.addWidget(self.clearButton)
        menuLayout.addWidget(self.abortButton)

        layout = QVBoxLayout()
        layout.addLayout(menuLayout)
        layout.addWidget(self.signatureCanvas)
        self.setLayout(layout)

    def verify(self):
        ...
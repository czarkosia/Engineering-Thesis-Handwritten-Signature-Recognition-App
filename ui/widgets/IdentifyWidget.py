from PyQt6.QtCore import pyqtSignal, QSize, QRect
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QSizePolicy

from ui.widgets.SignatureCanvasWidget import SignatureCanvasWidget
from ui.config import DATA_DIR

class IdentifyWidget(QWidget):
    goToStart = pyqtSignal()
    def __init__(self):
        super(IdentifyWidget, self).__init__()
        self.tempLabel = QLabel("Nothing's here yet")
        self.signatureCanvas = SignatureCanvasWidget()
        self.signatureCanvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.signatureCanvas.clear)

        self.abortButton = QPushButton("Abort")
        self.abortButton.clicked.connect(self.goToStart.emit)

        layout = QHBoxLayout()
        layout.addWidget(self.signatureCanvas)
        layout.addWidget(self.clearButton)
        layout.addWidget(self.abortButton)
        self.setLayout(layout)

    def identify(self):
        try:
            # self.signatureCanvas.save(DATA_DIR / "identify.txt")
            self.signatureCanvas.clear()
            ...
        except Exception as e:
            print("Failed to save signature data for identification: ", e)


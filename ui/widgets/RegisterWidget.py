from PyQt6.QtCore import pyqtSignal, Qt, QRect
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog, QLineEdit, QStackedLayout, \
    QLabel, QSizePolicy
from shutil import rmtree

from ui.config import REGISTERED_DATA_DIR
from ui.widgets.SignatureCanvasWidget import SignatureCanvasWidget

class RegisterWidget(QWidget):
    goToStart = pyqtSignal()

    def __init__(self, parent=None):
        super(RegisterWidget, self).__init__(parent)
        self.registered = 0

        self.signatureCanvas = SignatureCanvasWidget()
        self.signatureCanvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.nameBox = QLineEdit()
        self.nameLabel = QLabel("Name:")
        self.nameLabel.setBuddy(self.nameBox)

        self.registerButton = QPushButton("Register")
        self.registerButton.clicked.connect(self.register)
        self.registeredLabel = QLabel(str(self.registered) + "/5")
        self.registeredLabel.setBuddy(self.registerButton)


        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.signatureCanvas.clear)

        self.abortButton = QPushButton("Abort")
        self.abortButton.clicked.connect(self.abort)

        self.menuWidget = QWidget()
        menuLayout = QHBoxLayout()
        menuLayout.addWidget(self.nameLabel)
        menuLayout.addWidget(self.nameBox)
        menuLayout.addWidget(self.registerButton)
        menuLayout.addWidget(self.registeredLabel)
        menuLayout.addWidget(self.clearButton)
        menuLayout.addWidget(self.abortButton)
        self.menuWidget.setLayout(menuLayout)

        self.globalLayout = QVBoxLayout()
        self.globalLayout.addWidget(self.menuWidget)
        self.globalLayout.addWidget(self.signatureCanvas)
        self.setLayout(self.globalLayout)

    def register(self):
        if self.nameBox.text() != "":
            try:
                dir = (REGISTERED_DATA_DIR / self.nameBox.text())
                if not dir.exists():
                    dir.mkdir(parents=True)
                self.signatureCanvas.save(dir / f"{self.nameBox.text()}_{str(self.registered)}.txt")
                self.registered += 1
                self.registeredLabel.setText(str(self.registered) + "/5")
                self.nameBox.setEnabled(False)
                if self.registered >= 5:
                    print("Registration successful")
                    self.registerButton.setEnabled(False)

            except Exception as e:
                print("Failed to register: " + str(e))
        else:
            print("Name can't be empty")

    def abort(self):
        if self.registered < 5:
            print("Registration unsuccessful")
            rmtree(REGISTERED_DATA_DIR / self.nameBox.text())

        self.nameBox.setEnabled(True)
        self.nameBox.setText("")

        self.registerButton.setEnabled(True)
        self.registeredLabel.setText("0/5")
        self.registered = 0

        self.goToStart.emit()

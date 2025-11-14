from PIL.ImageQt import QPixmap
from PyQt6.QtCore import pyqtSignal, QRect
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy, QGraphicsColorizeEffect


class StartWidget(QWidget):
    goToRegister = pyqtSignal()
    goToVerify = pyqtSignal()
    goToIdentify = pyqtSignal()

    def __init__(self, parent=None):
        super(StartWidget, self).__init__(parent)
        self.register_button = QPushButton("Register")
        self.register_button.setObjectName("register_button")
        self.register_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.register_button.clicked.connect(self.goToRegister.emit)

        self.verify_button = QPushButton("Verify")
        self.verify_button.setObjectName("verify_button")
        self.verify_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.verify_button.clicked.connect(self.goToVerify.emit)

        self.identify_button = QPushButton("Identify")
        self.identify_button.setObjectName("identify_button")
        self.identify_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.identify_button.clicked.connect(self.goToIdentify.emit)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.register_button)
        self.layout().addWidget(self.verify_button)
        self.layout().addWidget(self.identify_button)

        pixmap = QPixmap("ui/styles/images/start-buttons.png")
        slice_top = pixmap.height() / 3
        slice_bottom = pixmap.height() / 3 * 2
        qss = f'''
        QPushButton {{
            color: #fff;
            font-size: 64px;
                
        }}
        QPushButton#register_button {{
            border: none;
            border-image: url("ui/styles/images/start-buttons.png") 0 0 {slice_bottom} 0;
        }} 
        QPushButton#verify_button {{
            border: none;
            border-image: url("ui/styles/images/start-buttons.png") {slice_top} 0 {slice_top} 0;
        }}
        QPushButton#identify_button {{
            border: none;
            border-image: url("ui/styles/images/start-buttons.png") {slice_bottom} 0 0 0 stretch stretch;
        }}
        QPushButton:hover {{
            filter: brightness(0.7);
        }}    
        '''
        self.setStyleSheet(qss)

from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QTabletEvent, QPainter, QColor
from PyQt6.QtWidgets import QWidget


class SignatureCanvasWidget(QWidget):
    def __init__(self):
        super(SignatureCanvasWidget, self).__init__()
        self.points = []
        self.palette().setColor(self.backgroundRole(), QColor(0, 0, 0))
        self.show()

    def tabletEvent(self, event: QTabletEvent):
        print("tabletEvent")
        event.accept()

    def clear(self):
        self.points = []
        print("clear")
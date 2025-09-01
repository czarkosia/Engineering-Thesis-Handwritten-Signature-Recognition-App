import time

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QTabletEvent, QPainter, QColor, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget
from typing_extensions import override


class SignatureCanvasWidget(QWidget):
    def __init__(self):
        super(SignatureCanvasWidget, self).__init__()
        self.start_time = time.time()
        self.x_coords, self.y_coords = [], []
        self.pressures, self.timestamps = [], []
        self.show()

    def tabletEvent(self, event):
        if event.type() in (QEvent.Type.TabletMove, QEvent.Type.TabletPress):
            print('tabletEvent')
            coords = event.position()
            self.x_coords.append(coords.x())
            self.y_coords.append(coords.y())
            self.pressures.append(event.pressure())
            self.timestamps.append(time.time() - self.start_time)

        self.update()
        event.accept()

    def paintEvent(self, event: QPaintEvent):
        print('paintEvent')
        if (self.y_coords == []
                or self.x_coords == []
                or self.pressures == []
                or self.timestamps == []):
            return

        painter = QPainter(self)
        pen = QPen()
        pen.setColor(QColor.fromRgb(0,255,0))
        prev_x, prev_y = self.x_coords[0], self.y_coords[0]
        for x, y, pressure in zip(self.x_coords, self.y_coords, self.pressures):

            pen.setWidthF(1 + 5 * pressure)
            painter.setPen(pen)
            painter.drawPoint(int(x), int(y))
            prev_x = x
            prev_y = y

    def clear(self):
        print("clear")

        print('X coords: ', self.x_coords)
        print('Y coords: ', self.y_coords)
        print('Pressures: ', self.pressures)
        print('Timestamps: ', self.timestamps)

        self.x_coords, self.y_coords = [], []
        self.pressures, self.timestamps = [], []
        self.update()

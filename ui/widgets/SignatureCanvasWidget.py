import time

from PyQt6.QtCore import QEvent, QSize
from PyQt6.QtGui import QPainter, QColor, QPaintEvent, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

class SignatureCanvasWidget(QWidget):
    def __init__(self):
        super(SignatureCanvasWidget, self).__init__()
        self.start_time = time.time()
        self.x_coords, self.y_coords = [], []
        self.pressures, self.timestamps = [], []
        self.setFixedSize(QSize(640, 480))

        self.buffer = QPixmap(self.size())
        self.buffer.fill(QColor(255, 255, 255))

    def resizeEvent(self, event):
        if self.size() != self.buffer.size():
            new_buffer = QPixmap(self.size())
            new_buffer.fill(QColor("white"))
            painter = QPainter(new_buffer)
            painter.drawPixmap(0, 0, self.buffer)
            painter.end()
            self.buffer = new_buffer
        super().resizeEvent(event)

    def tabletEvent(self, event):
        if event.type() in (QEvent.Type.TabletMove, QEvent.Type.TabletPress):
            print('tabletEvent')
            coords = event.position()
            self.x_coords.append(coords.x())
            self.y_coords.append(coords.y())
            self.pressures.append(event.pressure())
            self.timestamps.append(time.time() - self.start_time)

            painter = QPainter(self.buffer)
            pen = QPen()
            pen.setWidthF(1 + 5 * event.pressure())
            painter.setPen(pen)
            x = int(self.x_coords[-1])
            y = int(self.y_coords[-1])
            if len(self.timestamps) > 1 and self.timestamps[-1] - self.timestamps[-2] < 0.01:
                prevX = int(self.x_coords[-2])
                prevY = int(self.y_coords[-2])
                painter.drawLine(prevX, prevY, x, y)
            else:
                painter.drawPoint(x, y)
            painter.end()

            self.update()
            event.accept()

    def paintEvent(self, event: QPaintEvent):
        print('paintEvent')
        painter = QPainter(self)
        if not self.buffer.isNull() and self.buffer.width() > 0 and self.buffer.height() > 0:
            painter.drawPixmap(0, 0, self.buffer)
        painter.end()

    def save (self):
        if (not self.x_coords
                or not self.y_coords
                or not self.pressures
                or not self.timestamps):
            raise Exception("Missing points data to save")
        if not (len(self.x_coords) == len(self.y_coords) == len(self.pressures) == len(self.timestamps)):
            raise Exception("Mismatch between points features")

        # with open(".txt", "a") as f:
        #     for i in range(len(self.x_coords)):
        #         x = int(self.x_coords[i])
        #         y = int(self.y_coords[i])
        #         line = str(x) + "," + str(y)
        #         f.write(line)
        self.clear()

    def clear(self):
        print("clear")

        print('X coords: ', self.x_coords)
        print('Y coords: ', self.y_coords)
        print('Pressures: ', self.pressures)
        print('Timestamps: ', self.timestamps)

        self.x_coords, self.y_coords = [], []
        self.pressures, self.timestamps = [], []
        self.buffer.fill(QColor(255, 255, 255))
        self.update()

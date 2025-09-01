from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QPushButton

from gui.widgets.signature_canvas import SignatureCanvasWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Handwritten Signature Recognition")
        screen = QApplication.primaryScreen().size()
        width, height = screen.width(), screen.height()
        self.setGeometry(int(width/2) - 400, int(height/2) - 300, 800, 600)

        self.signature_canvas = SignatureCanvasWidget()
        self.button = QPushButton("Click me")
        self.button.clicked.connect(self.signature_canvas.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.signature_canvas)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # def create_menus(self):
        # fileMenu = QtWidgets.QMenuBar().addMenu(tr("File"))
        # fileMenu->addAction(tr("&Open..."), QKeySequence::Open, self.load)
        # fileMenu->addAction(tr("&Save As..."), QKeySequence::SaveAs, self.save)
        # fileMenu->addAction(tr("&New"), QKeySequence::New, self.clear)
        # fileMenu->addAction(tr("E&xit"), QKeySequence::Quit, self.close)
        # brushMenu = menuBar().addMenu(tr("Brush"))
        # brushMenu->addAction(tr("&Brush Color..."), tr("Ctrl+B"), self.setBrushColor)
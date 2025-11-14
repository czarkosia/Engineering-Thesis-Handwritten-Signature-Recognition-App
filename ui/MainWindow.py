from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QStackedWidget

from ui.widgets import RegisterWidget, StartWidget, VerifyWidget, IdentifyWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Handwritten Signature Recognition")
        screen = QApplication.primaryScreen().size()
        width, height = screen.width(), screen.height()
        self.setGeometry(int(width/2) - 400, int(height/2) - 300, 800, 600)
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.stackedWidget = QStackedWidget(self.mainWidget)

        layout = QVBoxLayout(self.mainWidget)
        self.stackedWidget = QStackedWidget()
        layout.addWidget(self.stackedWidget)

        self.startWidget = StartWidget()
        self.stackedWidget.addWidget(self.startWidget)
        self.stackedWidget.setCurrentWidget(self.startWidget)
        self.startWidget.goToRegister.connect(self.switch_to_register)
        self.startWidget.goToVerify.connect(self.switch_to_verify)
        self.startWidget.goToIdentify.connect(self.switch_to_identify)

        self.registerWidget = RegisterWidget()
        self.stackedWidget.addWidget(self.registerWidget)
        self.registerWidget.goToStart.connect(self.switch_to_start)

        self.verifyWidget = VerifyWidget()
        self.stackedWidget.addWidget(self.verifyWidget)
        self.verifyWidget.goToStart.connect(self.switch_to_start)

        self.identifyWidget = IdentifyWidget()
        self.stackedWidget.addWidget(self.identifyWidget)
        self.identifyWidget.goToStart.connect(self.switch_to_start)

    def switch_to_register(self):
        self.stackedWidget.setCurrentWidget(self.registerWidget)
        print("Switched to Register")

    def switch_to_start(self):
        self.stackedWidget.setCurrentWidget(self.startWidget)
        print("Switched to Start")

    def switch_to_verify(self):
        self.stackedWidget.setCurrentWidget(self.verifyWidget)
        print("Switched to Verify")

    def switch_to_identify(self):
        self.stackedWidget.setCurrentWidget(self.identifyWidget)
        print("Switched to Identify")

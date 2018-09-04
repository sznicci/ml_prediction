from src.DialogForTab1 import *
from src.DialogForTab2 import *
import sys


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Bike station status'
        self.left = 50
        self.top = 80
        self.width = 700
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = BikeStationStatusTabs(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class BikeStationStatusTabs(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.dialogForTab1 = DialogForTab1()
        self.dialogForTab2 = DialogForTab2()

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Train the machine")
        self.tabs.addTab(self.tab2, "Predict status")

        # Create tab1
        self.tab1.layout = QFormLayout()
        self.tab1.layout.addWidget(self.dialogForTab1)
        self.tab1.setLayout(self.tab1.layout)

        # Create tab2
        self.tab2.layout = QFormLayout()
        self.tab2.layout.addWidget(self.dialogForTab2)
        self.tab2.setLayout(self.tab2.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

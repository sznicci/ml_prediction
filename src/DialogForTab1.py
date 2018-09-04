from PyQt5.QtWidgets import *
from src.TrainModel import *
from src.DialogForTab2 import *
from src.Gui import *
import pandas as pd


class DialogForTab1(QDialog):
    NumGridRows = 2
    NumButtons = 2

    def __init__(self):
        super(DialogForTab1, self).__init__()
        self.inputFileName = QLineEdit()
        self.formGroupBox = QGroupBox()
        self.createFormGroupBox()
        self.selectButton = self.createSelectButton()
        self.trainButton = self.createTrainButton()

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formGroupBox)
        self.mainLayout.addWidget(self.selectButton)
        self.mainLayout.addWidget(self.trainButton)
        self.setLayout(self.mainLayout)

        self.setWindowTitle("File for training")

    def createSelectButton(self):
        selectButton = QPushButton("Select file", self)
        selectButton.clicked.connect(self.clickSelect)
        return selectButton

    def createTrainButton(self):
        trainButton = QPushButton("Train", self)
        trainButton.clicked.connect(self.clickTrain)
        return trainButton

    def clickSelect(self):
        self.showOpenFileDialog()

    def showOpenFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getOpenFileName(self, 'Open file', "/home", "CSV files (*.csv);;All Files (*)",
                                               options=options)
        self.inputFileName.setText(fileName[0])

    def clickTrain(self):
        print(self.inputFileName.text())
        df = pd.read_csv(self.inputFileName.text())
        trainAndSaveAll(df)
        self.mainLayout.addWidget(QLabel("Success"))
        self.mainLayout.removeWidget(self.selectButton)
        self.mainLayout.removeWidget(self.trainButton)
        print("Train")

    def createFormGroupBox(self):
        layout = QFormLayout()
        layout.addRow(QLabel("Select a .csv file for the training."), self.inputFileName)
        self.formGroupBox.setLayout(layout)

from src.Gui import *
import pandas as pd
from src.TrainModel import trainAndSaveAll


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

        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formGroupBox)
        self.mainLayout.addWidget(self.selectButton)
        self.mainLayout.addWidget(self.trainButton)
        self.mainLayout.addWidget(self.progressBar)
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
        self.onStart()
        print(self.inputFileName.text())
        df = pd.read_csv(self.inputFileName.text())
        trainAndSaveAll(df)
        self.onFinished()
        self.mainLayout.addWidget(QLabel("Success"))
        print("Train")

    def onStart(self):
        self.progressBar.setValue(50)

    def onFinished(self):
        # Stop the pulsation
        self.progressBar.setValue(100)

    def createFormGroupBox(self):
        layout = QFormLayout()
        layout.addRow(QLabel("Select a .csv file for the training."), self.inputFileName)
        self.formGroupBox.setLayout(layout)

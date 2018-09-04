from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from src.encodeData import *
import pickle


class DialogForTab2(QDialog):

    def __init__(self):
        super(DialogForTab2, self).__init__()
        self.createFormGroupBox()

        self.predictButton = QPushButton("Predict", self)
        self.predictButton.clicked.connect(self.clickPredict)

        self.predictionTextArea = QTextEdit()
        self.predictionTextArea.resize(300, 300)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formGroupBox)
        self.mainLayout.addWidget(self.predictButton)
        self.mainLayout.addWidget(self.predictionTextArea)

        self.stationComboBox()
        self.setLayout(self.mainLayout)

    def clickPredict(self):
        values = pd.DataFrame([(int(self.getCurrentStation()),
                                encoder_time.transform(pd.DataFrame([self.getTime().toString('HH:mm:00')])
                                                       .values.ravel().astype(str)),
                                int(self.getDateTime().toString('dd')), int(self.getDateTime().toString('MM')),
                                encoder_tempMax.transform(pd.DataFrame([self.getTempRange()])
                                                          .values.ravel().astype(str)), int(self.getTemperature()),
                                encoder_tempMin.transform(pd.DataFrame([self.getMinTempRange()])
                                                          .values.ravel().astype(str)), int(self.getTemperature()),
                                encoder_rain.transform(pd.DataFrame([self.rains.currentText()])
                                                       .values.ravel().astype(str)),
                                self.getRain(),
                                self.getWindSpeed(self.winds.currentText()))])

        print(values)
        self.predictBoth(values)

    def predictBoth(self, values):
        filenameDtree = 'finalized_dtree.sav'
        filenameKnn = 'finalized_knn.sav'
        filenameClf = 'finalized_clf.sav'
        loaded_dtree = pickle.load(open(filenameDtree, 'rb'))
        loaded_knn = pickle.load(open(filenameKnn, 'rb'))
        loaded_clf = pickle.load(open(filenameClf, 'rb'))

        dTreePredictionTakeout = self.convertStatusToString(loaded_dtree.predict(values)[0])
        knnPredictionTakeout = self.convertStatusToString(loaded_knn.predict(values)[0])
        nnPredictionTakeout = self.convertStatusToString(loaded_clf.predict(values)[0])
        dTreePredictionReturn = self.exchangeFromTakeoutToReturn(dTreePredictionTakeout)
        knnPredictionReturn = self.exchangeFromTakeoutToReturn(knnPredictionTakeout)
        nnPredictionReturn = self.exchangeFromTakeoutToReturn(nnPredictionTakeout)
        print("predicted takeouts dtree ", dTreePredictionTakeout)
        print("predicted returns dtree ", dTreePredictionReturn)
        print("predicted takeouts knn ", knnPredictionTakeout)
        print("predicted returns knn ", knnPredictionReturn)
        print("predicted takeouts clf ", nnPredictionTakeout)
        print("predicted returns clf ", nnPredictionReturn)

        print("Predicted status for station {} on day {} and month {}".format(values[0].get(0), values[2].get(0),
                                                                              values[3].get(0)))

        self.predictionTextArea.append("\nPredicted status for station {} on day {} and month {}"
                                       .format(values[0].get(0), values[2].get(0), values[3].get(0)))

        header = "\tTakeout \t\t Return"
        # self.predictionTextArea.insertHtml(text)
        self.predictionTextArea.append(header)
        predictionDT = "Decision tree: " + str(dTreePredictionTakeout) + "\t\tDecision tree: " \
                       + str(dTreePredictionReturn)
        predictionkNN = "k-nearest neighbor: " + str(knnPredictionTakeout) + "\t\tk-nearest neighbor: " \
                        + str(knnPredictionReturn)
        predictionNN = "Neural network: " + str(nnPredictionTakeout) + "\t\tNeural network: " + str(nnPredictionReturn)
        self.predictionTextArea.append(predictionDT)
        self.predictionTextArea.append(predictionkNN)
        self.predictionTextArea.append(predictionNN)
       
        print("predict")

    @staticmethod
    def exchangeFromTakeoutToReturn(takeout):
        if takeout == 'green':
            return 'red'
        elif takeout == 'blue':
            return 'amber'
        elif takeout == 'amber':
            return 'blue'
        elif takeout == 'red':
            return 'green'

    def convertStatusToString(self, takeout):
        if takeout == 1:
            return 'red'
        elif takeout == 2:
            return 'amber'
        elif takeout == 3:
            return 'blue'
        elif takeout == 4:
            return 'green'

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Predict station status")
        self.stations = self.stationComboBox()
        self.date = self.dateTimeWidget()
        self.time = self.timeWidget()
        self.temperatures = self.temperature()
        self.rains = self.rain()
        self.winds = self.winds()
        layout = QFormLayout()
        layout.addRow(QLabel("Station"), self.stations)
        layout.addRow(QLabel("Date"), self.date)
        layout.addRow(QLabel("Time"), self.time)
        layout.addRow(QLabel("Temperature"), self.temperatures)
        layout.addRow(QLabel("Rain"), self.rains)
        layout.addRow(QLabel("Wind"), self.winds)
        self.formGroupBox.setLayout(layout)

    def rain(self):
        comboBoxRain = QComboBox()
        comboBoxRain.addItem("dry")
        comboBoxRain.addItem("drizzle")
        comboBoxRain.addItem("moderate rain")
        comboBoxRain.addItem("heavy rain")
        comboBoxRain.currentIndexChanged.connect(self.getRain)
        return comboBoxRain

    def getRain(self):
        if self.rains.currentText() == 'dry':
            return 0
        elif self.rains.currentText() == 'drizzle':
            return 0.3
        elif self.rains.currentText() == 'moderate rain':
            return 2.25
        elif self.rains.currentText() == 'heavy rain':
            return 13.05

    def winds(self):
        comboBoxWind = QComboBox()
        comboBoxWind.addItem("calm")
        comboBoxWind.addItem("light air")
        comboBoxWind.addItem("breeze")
        comboBoxWind.addItem("strong wind")
        comboBoxWind.currentIndexChanged.connect(self.getWindSpeed)
        return comboBoxWind

    @staticmethod
    def getWindSpeed(wind):
        if wind == 'calm':  # 0
            return 0
        elif wind == 'light air':  # 1-3
            return 2
        elif wind == 'breeze':  # 4 <= wind <= 27
            return 15
        elif wind == 'strong wind':  # 28 <= wind <= 40
            return 32

    def temperature(self):
        comboBoxTemp = QComboBox()
        for temp in range(-10, 40):
            if temp == 25:
                comboBoxTemp.setCurrentIndex(temp)
            comboBoxTemp.addItem(str(temp))
        comboBoxTemp.currentIndexChanged.connect(self.getTemperature)
        return comboBoxTemp

    def getTemperature(self):
        return self.temperatures.currentText()

    def getTempRange(self):
        if int(self.temperatures.currentText()) < 0:
            return "0-5C"
        if 0 <= int(self.temperatures.currentText()) < 5:
            return "0-5C"
        elif 5 <= int(self.temperatures.currentText()) < 10:
            return ">=5-<10C"
        elif 10 <= int(self.temperatures.currentText()) < 15:
            return ">=10C-<15C"
        elif 15 <= int(self.temperatures.currentText()) < 20:
            return ">=15C-<20C"
        elif 20 <= int(self.temperatures.currentText()) < 25:
            return ">=20C-<25C"
        elif 25 <= int(self.temperatures.currentText()):
            return ">=25C"

    def getMinTempRange(self):
        if int(self.temperatures.currentText()) < 0:
            return "<0C"
        elif 0 <= int(self.temperatures.currentText()) < 5:
            return "0-5C"
        elif 5 <= int(self.temperatures.currentText()) < 10:
            return ">=5C-<10C"
        elif 10 <= int(self.temperatures.currentText()) < 15:
            return ">=10C-<15C"
        elif 15 <= int(self.temperatures.currentText()) < 20:
            return ">=15C-<20C"
        elif 20 <= int(self.temperatures.currentText()) < 25:
            return ">=20C-<25C"
        elif 25 <= int(self.temperatures.currentText()):
            return ">=25C"

    @staticmethod
    def dateTimeWidget():
        datePicker = QDateTimeEdit(QDate.currentDate())
        datePicker.setCalendarPopup(True)
        datePicker.setDisplayFormat('dd MMMM')
        return datePicker

    def getDateTime(self):
        return self.date.dateTime()

    @staticmethod
    def timeWidget():
        timePicker = QTimeEdit(QTime.currentTime())
        timePicker.setCalendarPopup(True)
        timePicker.setDisplayFormat('HH:mm')
        return timePicker

    def getTime(self):
        return self.time.time()

    def stationComboBox(self):
        comboBox = QComboBox()
        stations = (["7", "Charlbert Street, St. John's Wood", "small"],
                    ["101", "Queen Street 1, Bank", "underground"],
                    ["116", "Little Argyll Street, West End", "underground"],
                    ["170", "Hardwick Street, Clerkenwell", "other"],
                    ["240", "Colombo Street, Southwark", "underground"],
                    ["251", "Brushfield Street, Liverpool Street", "train"],
                    ["254", "Chadwell Street, Angel", "small"],
                    ["301", "Marylebone Lane, Marylebone", "underground"],
                    ["324", "Ontario Street, Elephant & Castle", "small"],
                    ["330", "Eastbourne Mews, Paddington", "train"],
                    ["410", "Edgware Road Station, Marylebone", "underground"],
                    ["435", "Kennington Station, Kennington", "other"],
                    ["467", "Southern Grove, Bow", "small"],
                    ["501", "Cephas Street, Bethnal Green", "other"],
                    ["679", "Orbel Street, Battersea", "other"],
                    ["693", "Felsham Road, Putney", "other"],
                    ["722", "Finnis Street, Bethnal Green", "small"],
                    ["732", "Duke Street Hill, London Bridge", "train"],
                    ["784", "East Village, Queen Elizabeth Olympic Park", "train"],
                    ["798", "Birkenhead Street, King's Cross", "train"])
        for station in stations:
            comboBox.addItem(station[0] + " - " + station[1] + " - " + station[2])
        comboBox.currentIndexChanged.connect(self.getCurrentStation)
        return comboBox

    def getCurrentStation(self):
        return self.stations.currentText().split(" - ")[0]

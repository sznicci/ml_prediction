from src.Predict import *
import pandas as pd
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pickle


def trainAndSaveAll(dft):
    encode_rainLabels(dft)
    encode_tempMaxLabels(dft)
    encode_tempMinLabels(dft)
    encode_timeLabels(dft)

    # Set input data
    Xt = dft.drop(['status_takeouts', 'status_returns'], axis=1)

    # Set output data
    yt = dft['status_takeouts']

    # Split the data into a training set and a test set
    X_traint, X_testt, y_traint, y_testt = train_test_split(Xt, yt, test_size=0.3)

    # train decision tree
    trainDF(X_traint, y_traint, X_testt, y_testt)

    # train kNN
    trainKNN(X_traint, y_traint, X_testt, y_testt)

    # train NN
    trainNN(X_traint, y_traint, X_testt, y_testt)


def trainDF(X_traint, y_traint, X_testt, y_testt):
    # Instantiate a decision tree classifier
    dtreet = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5)

    dtreet.fit(X_traint, y_traint)

    # save the model to disk
    filenameDF = 'finalized_dtree.sav'
    pickle.dump(dtreet, open(filenameDF, 'wb'))

    print("DT")
    predict(dtreet, X_testt, y_testt)


def trainKNN(X_traint, y_traint, X_testt, y_testt):
    h = .02  # step size in the mesh

    knn = neighbors.KNeighborsClassifier()

    # we create an instance of Neighbours Classifier and fit the data.
    knn.fit(X_traint, y_traint)

    # save the model to disk
    filenameKNN = 'finalized_knn.sav'
    pickle.dump(knn, open(filenameKNN, 'wb'))

    print("kNN")
    predict(knn, X_testt, y_testt)


def trainNN(X_traint, y_traint, X_testt, y_testt):
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(50, 50, 30), random_state=1)

    clf.fit(X_traint, y_traint)

    # save the model to disk
    filenameNN = 'finalized_clf.sav'
    pickle.dump(clf, open(filenameNN, 'wb'))

    print("NN")
    predict(clf, X_testt, y_testt)


# Encode string labels to categories
def encode_timeLabels(dft):
    le_time = preprocessing.LabelEncoder()
    timeList = pd.timedelta_range(start='00:00:00', end='23:59:00', freq="1T")
    le_time.fit(timeList.astype(str))
    dft['time'] = le_time.transform(dft['time'].astype(str))


def encode_rainLabels(dft):
    le_rain = preprocessing.LabelEncoder()

    le_rain.fit(dft['rain'])
    dft['rain'] = le_rain.transform(dft['rain'].astype(str))


def encode_tempMaxLabels(dft):
    le_temp_max = preprocessing.LabelEncoder()

    le_temp_max.fit(dft['temp_max'])
    dft['temp_max'] = le_temp_max.transform(dft['temp_max'].astype(str))


def encode_tempMinLabels(dft):
    le_temp_min = preprocessing.LabelEncoder()

    le_temp_min.fit(dft['temp_min'])
    dft['temp_min'] = le_temp_min.transform(dft['temp_min'].astype(str))

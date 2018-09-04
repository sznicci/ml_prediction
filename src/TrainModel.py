from src.encodeData import *
from src.Predict import *
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pickle


def trainAndSaveAll(dft):
    dft['rain'] = encoder_rain.transform(dft['rain'].astype(str))
    dft['temp_max'] = encoder_tempMax.transform(dft['temp_max'].astype(str))
    dft['temp_min'] = encoder_tempMin.transform(dft['temp_min'].astype(str))
    dft['time'] = encoder_time.transform(dft['time'].astype(str))

    # Set input data
    Xt = dft.drop('status_takeouts', axis=1)

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

    predict(dtreet, X_testt, y_testt)


def trainKNN(X_traint, y_traint, X_testt, y_testt):
    h = .02  # step size in the mesh

    knn = neighbors.KNeighborsClassifier()

    # we create an instance of Neighbours Classifier and fit the data.
    knn.fit(X_traint, y_traint)

    # save the model to disk
    filenameKNN = 'finalized_knn.sav'
    pickle.dump(knn, open(filenameKNN, 'wb'))

    predict(knn, X_testt, y_testt)


def trainNN(X_traint, y_traint, X_testt, y_testt):
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(50, 50, 30), random_state=1)

    clf.fit(X_traint, y_traint)

    # save the model to disk
    filenameNN = 'finalized_clf.sav'
    pickle.dump(clf, open(filenameNN, 'wb'))

    predict(clf, X_testt, y_testt)

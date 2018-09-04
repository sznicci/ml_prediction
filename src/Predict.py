import pandas as pd
import seaborn
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


def predict(decisionTree, X_test, y_test):
    predictions = decisionTree.predict(X_test)

    conf_matrix = confusion_matrix(y_test, predictions)

    print(conf_matrix)
    print(classification_report(y_test, predictions))

    print(predictions)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# import some data to play with
df = pd.read_csv('G:/total_day_month.csv')

# Encode string labels to categories
def encode_timeLabels():
    le_time = preprocessing.LabelEncoder()
    timeList = pd.timedelta_range(start='00:00:00', end='23:59:00', freq="1T")
    le_time.fit(timeList.astype(str))
    df['time'] = le_time.transform(df['time'].astype(str))
    return le_time


def encode_rainLabels():
    le_rain = preprocessing.LabelEncoder()

    le_rain.fit(df['rain'])
    df['rain'] = le_rain.transform(df['rain'].astype(str))
    return le_rain


def encode_tempMaxLabels():
    le_temp_max = preprocessing.LabelEncoder()

    le_temp_max.fit(df['temp_max'])
    df['temp_max'] = le_temp_max.transform(df['temp_max'].astype(str))
    return le_temp_max


def encode_tempMinLabels():
    le_temp_min = preprocessing.LabelEncoder()

    le_temp_min.fit(df['temp_min'])
    df['temp_min'] = le_temp_min.transform(df['temp_min'].astype(str))
    return le_temp_min


encoder_rain = encode_rainLabels()
encoder_tempMax = encode_tempMaxLabels()
encoder_tempMin = encode_tempMinLabels()
encoder_time = encode_timeLabels()

# Set input data
X = df.drop(['status_takeouts', 'status_returns'], axis=1)

# Set output data
Y = df['status_takeouts']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(20, 20, 20), random_state=1)

clf.fit(X_train, y_train)
#
# # save the model to disk
# filename = 'finalized_clf.sav'
# pickle.dump(clf, open(filename, 'wb'))

predictions = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)

print(conf_matrix)
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

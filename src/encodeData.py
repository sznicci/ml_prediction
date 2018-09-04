import pandas as pd
from sklearn import preprocessing


# Read the data from file
df = pd.read_csv('G:/total120k_day_month0.csv')


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

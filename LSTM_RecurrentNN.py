import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import timeit
import numpy

#function for preparation
def prepare_data(data, lags=1):
    """
    Create lagged data from an input time series
    """
    X, y = [], []
    for row in range(len(data) - lags - 1):
        a = data[row:(row + lags), 0]
        X.append(a)
        y.append(data[row + lags, 0])
    return np.array(X), np.array(y)

# fix random seed for reproducibility
np.random.seed(123)
# load the dataset
df = pd.read_csv('/home/alex/Desktop/doulke_mikri/ML/important_doc.txt', sep=',', parse_dates=True, index_col=0)
data = df.values
data = data.astype('float32')
# split into train and test sets
train = data[0:69, :]
test = data[70:, :]

# reshape into X=t and Y=t+1
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

### create and fit the LSTM network
t1=timeit.default_timer()
mdl = Sequential()
mdl.add(Dense(3, input_shape=(1, lags), activation='relu'))
mdl.add(LSTM(6, activation='relu'))
mdl.add(Dense(1, activation='relu'))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=2)
t2=timeit.default_timer()

### make predictions
test_predict = mdl.predict(X_test)

#evaluate predictions
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))

#find stats
real=test[4:]
predicted=test_predict
error=real-predicted
accuracy=100-abs((real-predicted)/real*100)
percent_error=100-accuracy
ma=sum(accuracy)/7
print("Model Accuracy:\n%s")%(ma)
print("Time consumed to train:\n%s ")%(t2-t1)
c=predicted
pec=percent_error


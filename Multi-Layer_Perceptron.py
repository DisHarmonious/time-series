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

###################Supervised Learning Task
# set seed
np.random.seed(756)
# import data set
df = pd.read_csv('/home/alex/Desktop/doulke_mikri/ML/important_doc.txt', sep=',', parse_dates=True, index_col=0)
data = df.values
# using keras often requires the data type float32
data = data.astype('float32')
# slice the data
train = data[0:71, :]  
test = data[72:, :]     

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
 
# prepare the data
lags = 1
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test     # due to naming convention

#MULTI-LAYER PERCEPTRON
# create and fit Multilayer Perceptron model
t1=timeit.default_timer()
mdl = Sequential()
mdl.add(Dense(3, input_dim=lags, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=1000, batch_size=2, verbose=2)
t2=timeit.default_timer()

# generate predictions for training
test_predict = mdl.predict(X_test)

#evaluate predictions
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))
 
#find stats
real=test[2:]
predicted=test_predict
error=real-predicted
accuracy=100-abs((real-predicted)/real*100)
percent_error=100-accuracy
ma=sum(accuracy)/7
print("Model Accuracy:\n%s")%(ma)
print("Time consumed to train:\n%s ")%(t2-t1)
b=predicted
peb=percent_error

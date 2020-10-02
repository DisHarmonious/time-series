from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import timeit
import pandas
import numpy

series = Series.from_csv('/home/alex/Desktop/doulke_mikri/ML/important_doc.txt', header=0)

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
t1=timeit.default_timer()
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
t2=timeit.default_timer()
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#find stats
real=test
predicted=predictions
error=real-predicted
accuracy=100-abs((real-predicted)/real*100)
percent_error=100-accuracy
ma=sum(accuracy)/7
print("Model Accuracy:\n%s")%(ma)
print("Time consumed to train:\n%s ")%(t2-t1)
a=predictions
pea=percent_error



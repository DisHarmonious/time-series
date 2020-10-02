import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##### mse #########
s = pd.Series(
    [2330475361, 647262016, 531775840],
    index = ['ARIMA', 'MLP(1000 epoch)', 'LSTMRNN(1000 epoch)']
)
plt.title("Mean Squared Error (MSE)")
#plt.ylabel('Time Consumed (s)')
plt.xlabel('Method')
ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = 'brg'  #red, green, blue, black, etc
s.plot( 
    kind='bar', 
    color=my_colors,
    rot=0
)
plt.show()

##### rmse ##############
s = pd.Series(
    [48274, 25441, 23060],
    index = ['ARIMA', 'MLP(1000 epoch)', 'LSTMRNN(1000 epoch)']
)
plt.title("Root Mean Squared Error (RMSE)")
#plt.ylabel('Time Consumed (s)')
plt.xlabel('Method')
ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = 'brg'  #red, green, blue, black, etc
s.plot( 
    kind='bar', 
    color=my_colors,
    rot=0
)
plt.show()

##### Each_percent_error ############
plt.plot(pea, color='blue', label='ARIMA')
plt.plot(peb, color='red', label='MLP(1000 epoch)')
plt.plot(pec, color='green', label='LSTMRNN(1000 epoch)')
plt.legend()
plt.xticks(arange(7), ('2011', '2012', '2013', '2014', '2015', '2016', '2017'))
plt.title('Percent deviation for each prediction')
plt.show()


##### Each prediction ############
plt.plot(a, color='blue', label='ARIMA')
plt.plot(b, color='red',label='MLP(1000 epoch)')
plt.plot(c, color='green',label='LSTMRNN(1000 epoch)')
plt.plot(real, color='yellow', label='Actual measurement')
plt.xticks(arange(7), ('2011', '2012', '2013', '2014', '2015', '2016', '2017'))
plt.legend()
plt.title('Graphical Comparison of Methods')
plt.show()

##### Accuracy #############
s = pd.Series(
    [85.28, 95.13, 94.62],
    index = ['ARIMA', 'MLP(1000 epoch)', 'LSTMRNN(1000 epoch)']
)
plt.title("Accuracy(%) of Methods")
#plt.ylabel('Time Consumed (s)')
plt.xlabel('Method')
ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = 'brg'  #red, green, blue, black, etc
s.plot( 
    kind='bar', 
    color=my_colors,
    rot=0
)
plt.show()

########## Epochs-Time #########
#mlp(100,500,1000): 5.751, 24.149, 47.774
#lstmrnn(100,500,1000): 20.363, 97.687, 184.625
n_groups = 3
mlp = (5.751, 24.149, 47.774)
lstmrnn = (20.363, 97.687, 184.625)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

rects1 = plt.bar(index, mlp, bar_width,
                 alpha=opacity,
                 color='b',
                 label='MLP')
rects2 = plt.bar(index + bar_width, lstmrnn, bar_width,
                 alpha=opacity,
                 color='r',
                 label='LSTMRNN')
plt.xlabel('# of Epochs')
plt.ylabel('Time (s)')
plt.title('Time consumed for n Epochs')
plt.xticks(index + bar_width / 2, ('n=100', 'n=500', 'n=1000'))
plt.legend()
plt.tight_layout()
plt.show()


########## Accuracy - Epochs ########
#mlp(100,500,1000): 94.48, 94.57, 95.13
#lstmrnn(100,500,1000): 90.98, 91.98, 94.62
n_groups = 3
mlp = (94.48, 94.57, 95.13)
lstmrnn = (90.98, 91.98, 94.62)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

rects1 = plt.bar(index, mlp, bar_width,
                 alpha=opacity,
                 color='b',
                 label='MLP')
rects2 = plt.bar(index + bar_width, lstmrnn, bar_width,
                 alpha=opacity,
                 color='r',
                 label='LSTMRNN')
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy(%)')
plt.title('Accuracy - Epochs')
plt.xticks(index + bar_width / 2, ('n=100', 'n=500', 'n=1000'))
plt.legend()
plt.tight_layout()
plt.show()







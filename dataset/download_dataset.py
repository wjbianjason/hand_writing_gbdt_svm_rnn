from keras.datasets import mnist 
import numpy as np
import pickle as pkl
(X_train,y_train),(X_test,y_test) = mnist.load_data()
f_pkl = open("classify/mnist.pkl",'wb')
#down sample
X_train = X_train[::5]
y_train = y_train[::5]
X_test = X_test[::5]
y_test = y_test[::5]

#convert the img size from (28,28) tot (14,14)
X_train_small = np.zeros(shape=(len(X_train),14,14))
X_test_small = np.zeros(shape=(len(X_test),14,14))
for index in range(len(X_train)):
	sample = X_train[index]
	for i in range(14):
		for j in range(14):
			mean_value = sum(sample[i*2:(i+1)*2,j*2:(j+1)*2].reshape(-1))/4
			X_train_small[index,i,j] = mean_value
for index in range(len(X_test)):
	sample = X_test[index]
	for i in range(14):
		for j in range(14):
			mean_value = sum(sample[i*2:(i+1)*2,j*2:(j+1)*2].reshape(-1))/4
			X_test_small[index,i,j] = mean_value
X_train_small = X_train_small.reshape((len(X_train_small),-1))
X_test_small = X_test_small.reshape((len(X_test_small),-1))


pkl.dump(((X_train_small,y_train),(X_test_small,y_test)),f_pkl)
f_pkl.close()



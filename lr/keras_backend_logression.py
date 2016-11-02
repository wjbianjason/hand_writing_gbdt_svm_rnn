from keras import backend as K
import numpy as np
import pickle as pkl
from keras.utils import np_utils
"""waiting list
(1)progress bar
"""
#################load data
DATA_PATH = "../dataset/classify/"
f_pkl = open(DATA_PATH+"mnist.pkl",'rb')
(x_train,y_train),(x_test,y_test) = pkl.load(f_pkl)
temp = x_train.std(axis=-1)
print temp.shape
x_train = (x_train - x_train.mean(axis=-1)[:,None]) / temp[:,None]
###########################################
def sigmoid(x):
	return 1 / (1+K.exp(x))
def softmax(x):
	return K.exp(x) / K.sum(K.exp(x),axis=-1)

y_old_label = y_test
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


batch_size = 16
epoch = 10
learning_rate = 0.1
nb_classes = 10
rng = np.random
print x_train.shape[1]
print y_train.shape[1]

x = K.placeholder((batch_size,x_train.shape[1]))
y = K.placeholder((batch_size,y_train.shape[1]))

#init weights
w = K.variable(rng.randn(nb_classes,x_train.shape[1]))
b = K.variable(rng.randn(nb_classes))

pred = softmax(-K.dot(x,w.transpose())-b)
# pred = K.softmax(-K.dot(x,w.transpose())-b)
pred_classes = pred.argmax(axis=-1)

xent = K.sum(-K.log(pred)*y,axis=-1)

l1_normal = 0.01 * (w ** 2).sum()

cost = xent.mean() + l1_normal	
gw,gb = K.gradients(cost,[w,b])

train = K.function(inputs=[x,y],outputs=[cost],updates=((w,w - learning_rate*gw),(b,b-learning_rate*gb)))
predict = K.function(inputs=[x],outputs=[pred_classes])

for i in range(epoch):
	print "epoch:",i
	for j in range(len(x_train)/batch_size):
		# print x_train[j*batch_size:(j+1)*batch_size].shape
		# print y_train[j*batch_size:(j+1)*batch_size].shape
		cost = train([x_train[j*batch_size:(j+1)*batch_size],y_train[j*batch_size:(j+1)*batch_size]])
		print cost[0]

pred_result = []
for j in range(len(x_test)/batch_size):
	pp = predict([x_test[j*batch_size:(j+1)*batch_size]])
	pred_result.extend(pp[0].tolist())
pred_result = np.asarray(pred_result)
print len(pred_result)
# exit(1)
accuracy = np.sum((pred_result == y_old_label)) / float(len(pred_result))
print accuracy
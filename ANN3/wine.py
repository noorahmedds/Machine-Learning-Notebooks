# from sklearn.model_selection import train_test_split
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy as np
#
# np.random.seed(3)
#
# # number of wine classes
# classifications = 3
#
# # load dataset
# dataset = np.loadtxt('wine.csv', delimiter=",")
#
# # split dataset into sets for testing and training
# X = dataset[:,1:14]
# Y = dataset[:,0:1]
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.66, random_state=5)
#
# # convert output values to one-hot
# y_train = keras.utils.to_categorical(y_train-1, classifications)
# y_test = keras.utils.to_categorical(y_test-1, classifications)
#
#
# # creating model
# model = Sequential()
# model.add(Dense(10, input_dim=13, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(classifications, activation='softmax'))
#
# # compile and fit model
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=15, epochs=2500, validation_data=(x_test, y_test))
#

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import genfromtxt

model = Sequential()

# now i create my layers
# the first hidden layer gets 13 features as input and it contains 10 nodes
hidden1 = Dense(10, input_dim=13, activation="relu")
hidden2 = Dense(8, activation="relu")
hidden3 = Dense(6, activation="relu")
hidden4 = Dense(6, activation="relu")
hidden5 = Dense(4, activation="relu")
# remember we dont need to add input dim to our output layers because in keras layers following first hidden layers infer their own input dims
outputLayer = Dense(3, activation="softmax")

# here we add our hidden layers to our model
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.add(hidden4)
model.add(hidden5)
model.add(outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


data = genfromtxt('wine.csv', delimiter=',')

train_ratio = 0.6

# split dataset into sets for testing and training
features = data[:,1:14]
labels = data[:,0:1]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=train_ratio, random_state=5)

# convert output values to one-hot
y_train = keras.utils.to_categorical(y_train-1, 3)
y_test = keras.utils.to_categorical(y_test-1, 3)


model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data = (x_test, y_test))

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# plt.imshow(x_train[1003])
y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)
# print(y_cat_test.shape)
# print(x_train[0])
x_train = x_train / 255
x_test = x_test / 255
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.show()

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)
# print(x_test.shape)

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(4, 4), input_shape=(x_train.shape[1:]),activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=["accuracy"])
model.summary()
model.fit(x_train,y_cat_train, epochs=2)
model.evaluate(x_test,y_cat_test)
predictions = model.predict(x_test)
predictions = np.argmax(predictions,1)
#plt.imshow(x_test[1,:,:,0])
#plt.show()
print(classification_report(y_test,predictions))
model.save("model.h5")
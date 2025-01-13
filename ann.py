import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten
from keras.layers import Dense

(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()

x_train_scaled = x_train/255
x_test_scaled = x_test/255

y_train_categorical = keras.utils.to_categorical(y_train,num_classes=10)
y_test_categorical = keras.utils.to_categorical(y_test,num_classes=10)


classes = ['aeroplane','automobile','bird','cat','dear','dog','frog','horse','ship','truck']

model = keras.Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(3000,activation='relu'))
model.add(Dense(10,activation='relu'))
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_scaled,y_train_categorical,epochs=2)
predictions = model.predict(x_test_scaled)
print(np.argmax(predictions[0]))
print(classes[np.argmax(predictions[0])])  
plt.imshow(x_test_scaled[0])
plt.show()
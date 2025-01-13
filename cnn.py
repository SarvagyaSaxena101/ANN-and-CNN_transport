import tensorflow as tf
import pandas as pd
import tf_keras 
import numpy as np
import matplotlib.pyplot as plt
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.layers import Flatten , Conv2D , MaxPooling2D
(x_train,y_train),(x_test,y_test) =  tf_keras.datasets.cifar10.load_data()

classes = ['aeroplan','automobile','bird','cat','dear','dog','frog','horse','ship','truck']

y_train = y_train.reshape(-1,)

x_train = x_train/255
x_test = x_test/255

model = Sequential() 

model.add(Conv2D(filters=32, kernel_size = (3,3),activation='relu',input_shape = (32,32,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters=64,kernel_size = (3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())
model.add(Dense(64,activation ='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs = 50)

print(model.evaluate(x_test,y_test))
predictions = model.predict(x_test)
print(classes[np.argmax(predictions[1])])
plt.imshow(x_test[1])
plt.show()
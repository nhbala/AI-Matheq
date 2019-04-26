import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import cv2

curr_url = "https://www.kaggle.com/hupe1980/keras-digit-recognizer-mnist-data"
def load_data():
    df_train  = pd.read_csv("train.csv")

    y_train = df_train['label'].values
    X_train = df_train.drop(columns=['label']).values


    return X_train, y_train

X_train, y_train = load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')

input_shape = X_train.shape[1:]

X_train = X_train

y_train = to_categorical(y_train)

num_classes = y_train.shape[1]

def convolutional_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

model = convolutional_model(num_classes)
model.fit(X_train, y_train, validation_split=0.1, epochs=4, batch_size=128, verbose=1)

im = cv2.imread('12.jpg',0)
im = cv2.resize(im,  (28, 28))
im.reshape((28,28)) # (28,28)

batch = np.expand_dims(im,axis=0)# (1, 28, 28)
batch = np.expand_dims(batch,axis=3) # (1, 28, 28,1)


print(model.predict(batch, batch_size=1))

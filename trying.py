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
from keras.models import model_from_yaml


yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights("model.h5")


im = cv2.imread('11.jpg',0)
im = cv2.resize(im,  (28, 28))
im.reshape((28,28))

batch = np.expand_dims(im,axis=0)
batch = np.expand_dims(batch,axis=3)

final = loaded_model.predict(batch, batch_size=1)
lmao = max(final[0])
result = [i for i, j in enumerate(final[0]) if j == lmao]
print(result)

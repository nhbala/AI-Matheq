import numpy as np
import pandas as pd
import matplotlib.pyplot as pt



data = pd.read_csv("train.csv").as_matrix()
print(data)
xtrain = data[0:21000, 1:]
train_label = data[0:21000, 0]
d=xtrain[8]
# print(d)

#
# new_np = np.array(d)
# final = new_np.reshape((28,28))
#
# print(final)
# pt.imshow(255-final, cmap='gray')
# pt.show()

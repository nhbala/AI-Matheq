import numpy as np
import pandas as pd
import matplotlib.pyplot as pt



data = pd.read_csv("train.csv").as_matrix()
xtrain = data[0:21000, :]
train_label = data[0:21000, 0]
print(xtrain)

# print(d)

#
# new_np = np.array(d)
# final = new_np.reshape((28,28))
#
# print(final)
# pt.imshow(255-final, cmap='gray')
# pt.show()

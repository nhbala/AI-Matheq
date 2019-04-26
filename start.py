import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import nn



data = pd.read_csv("train.csv").as_matrix()
xtrain = data[0:21000, :]
train_label = data[0:21000, 0]


curr_nn = nn.create_neural_network([784, 14])

nn.training_function(xtrain, curr_nn, 25, .25, nn.sigmoid_function, nn.sigmoid_function_derivative)
#print(results)
# print(d)

#
# new_np = np.array(d)
# final = new_np.reshape((28,28))
#
# print(final)
# pt.imshow(255-final, cmap='gray')
# pt.show()

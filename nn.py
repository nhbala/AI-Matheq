import numpy as np


#function takes in parameter layer_number_list which is a list of layers
#with their corresponding node number. Ex: [784, 100, 16, 14]
def create_neural_network(layer_number_list):
    final_lst = {}
    for i in range(len(layer_number_list)):
        final_lst["l" + str(i)] = np.zeros(layer_number_list[i])
        if i < len(layer_number_list) -1:
            final_lst["w" + str(i)] = np.random.uniform(-1, 1, (layer_number_list[i+1], layer_number_list[i]))
    return final_lst



def sigmoid_function(input):
    return 1/(1 + math.e**(-i))

def sigmoid_function_derivative(input):
    return sigmoid(input) * (1 - sigmoid_function(input))


def training_function(t_data, nn, run_amt, l_rate, ac_func, der_ac_func):
    x = 1

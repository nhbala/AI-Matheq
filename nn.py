import numpy as np


#function takes in parameter layer_number_list which is a list of layers
#with their corresponding node number. Ex: [784, 100, 16, 14]
def create_neural_network(layer_number_list):
    final_lst = {}
    for i in range(len(layer_number_list)):
        if i == 0:
            final_lst["l" + str(i)] = np.zeros(layer_number_list[i], 1)
        else:
            final_lst["l" + str(i)] = np.array([])
        if i < len(layer_number_list) -1:
            final_lst["w" + str(i)] = np.random.uniform(-1, 1, (layer_number_list[i+1], layer_number_list[i]))
    return final_lst

#mapping label to matrix representation
def map_label(symbol):
    matrix = np.zeros(14, 1)
    if symbol == "\d":
        matrix[eval(symbol)][0] = 1.0
    elif symbol == '+':
        matrix[10][0] = 1.0
    elif symbol == '-':
        matrix[11][0] = 1.0
    elif symbol == 'x':
        matrix[12][0] = 1.0
    elif symbol == '/':
        matrix[13][0] = 1.0
    return matrix


def sigmoid_function(input):
    return 1/(1 + math.e**(-i))

def sigmoid_function_derivative(input):
    return sigmoid(input) * (1 - sigmoid_function(input))


def training_function(t_data, nn, run_amt, l_rate, ac_func, der_ac_func):
    #iteration overall
    for x in range(run_amt):

        #looping through each example in training_data
        for t in range(len(t_data)):
            #forward algorithm


            #going through the initial node layer
            layer0 = nn["l0"]
            for n in range(len(t_data[t])-1):
                new_n = n + 1
                layer0[n] = t_data[t, new_n]
            nn["l0"] = layer0

            #saving a_j values for later
            inj_arr = []
            #set activation energy for other layers
            for i in (len(nn.getKeys()) -1)/2:
                vfunc = np.vectorize(ac_func)
                prev_layer = nn["l" + str(i)]
                weights = nn["w" + str(i)]
                new_layer_pre = np.matmul(weights, prev_layer)
                a_j_arr.append(new_layer_pre)
                next_layer = vfunc(new_layer_pre)
                nn["l" + str(i + 1)] = next_layer

            delta_j_arr = []
            #backward algorithm
            #finding delta_j values for output layer
            final_layer = nn["l" + (len(nn.getKeys())-1)
            inj_values = inj_arr.pop()
            for k in range(len(final_layer)):
                curr_inj = inj_values[k][0]
                delta_j = der_ac_func(curr_inj) * ((map_label(t_data[t][0]))[k][0] - final_layer[k][0])
                delta_j_arr.append(delta_j)

            #this is the second to last layer/weight number
            curr_start = ((len(nn.getKeys()) -1)/2) - 1
            for l in range(curr_start, -1, -1):
                curr_layer = nn["l" + l]
                cur_ini_values = inj_arr.pop()
                #need to work and finish this part
                for m in range(len(curr_layer)):
                    curr_ini = curr_ini_values[m][0]
                    delta_i = der_ac_func(curr_inj_values[m])










        for t in range(len(t_data)-1):
            new_t = t + 1
            if new_t == 1:
                first_layer = nn["l" + new_t]

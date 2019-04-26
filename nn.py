import numpy as np
import math


#function takes in parameter layer_number_list which is a list of layers
#with their corresponding node number. Ex: [784, 100, 16, 14]
def create_neural_network(layer_number_list):
    final_lst = {}
    for i in range(len(layer_number_list)):
        if i == 0:
            final_lst["l" + str(i)] = np.zeros((layer_number_list[i], 1))
        else:
            final_lst["l" + str(i)] = np.array([])
        if i < len(layer_number_list) -1:
            final_lst["w" + str(i)] = np.random.uniform(-1, 1, (layer_number_list[i+1], layer_number_list[i]))
    return final_lst

#mapping label to matrix representation
def map_label(symbol):
    matrix = np.zeros((14, 1))
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


def sigmoid_function(i):
    #print i
    #print math.e**(-i)
    return 1/(1 + math.e**(-i))

def sigmoid_function_derivative(i):
    return sigmoid_function(i) * (1 - sigmoid_function(i))


def training_function(t_data, nn, run_amt, l_rate, ac_func, der_ac_func):
    #iteration overall

    mse_lst = []
    for x in range(run_amt):
        total_mse = 0

        #looping through each example in training_data
        for t in range(len(t_data)):
            #forward algorithm


            #going through the initial node layer
            layer0 = nn["l0"]
            for n in range(len(t_data[t])-1):
                new_n = n + 1
                layer0[n] = t_data[t, new_n]/255
            nn["l0"] = layer0

            #saving inj values for later
            #a_j values are stored in neural network at each layer
            inj_arr = (nn['l0']).tolist()
            #set activation energy for other layers
            for i in range((len(nn.keys()) -1)/2):
                vfunc = np.vectorize(ac_func)
                prev_layer = nn["l" + str(i)]
                weights = nn["w" + str(i)]
                new_layer_pre = np.matmul(weights, prev_layer)
                inj_arr.append(new_layer_pre)
                next_layer = vfunc(new_layer_pre)
                nn["l" + str(i + 1)] = next_layer





            all_delta_j_arr = []
            delta_j_arr = []
            #backward algorithm
            #finding delta_j values for output layer
            final_layer = nn["l" + str((len(nn.keys()) -1)/2)]
            #print inj_arr
            inj_values = inj_arr.pop()
            mse_error = 0
            for k in range(len(final_layer)):
                curr_inj = inj_values[k][0]
                delta_j = der_ac_func(curr_inj) * ((map_label(t_data[t][0]))[k][0] - final_layer[k][0])
                mse_error += (delta_j ** 2)
                delta_j_arr.append(delta_j)
            all_delta_j_arr.append(delta_j_arr) #should we include delta J?
            print str(t) + ": " + str(mse_error)
            total_mse += mse_error



            #this is the layer number for second to last layer/last weight
            #print inj_arr
            curr_start = ((len(nn.keys()) -1)/2) - 1
            for l in range(curr_start, 0, -1): #right now stop before first layer, should we include?
                delta_i_arr = []
                curr_layer = nn["l" + str(l)]
                if l != 0:
                    curr_ini_values = inj_arr.pop()
                else:
                    curr_ini_values = inj_arr
                #Michael did the code from here pls check this over
                for m in range(len(nn['w' + str(l)])):
                    sum_weight_deltaj = 0
                    for p in range(len(delta_j_arr)):
                        sum_weight_deltaj = sum_weight_deltaj + nn["w" + str(l)][m][p] * delta_j_arr[p]
                    curr_ini = curr_ini_values[m][0]
                    delta_i = der_ac_func(curr_ini) * sum_weight_deltaj
                    delta_i_arr.append(delta_i)
                all_delta_j_arr.insert(0, delta_i_arr)
                delta_j_arr = delta_i_arr

            #update each weight
            for q in range(curr_start+1):
                curr_weights = nn["w" + str(q)]
                curr_layer = nn["l" + str(q)]
                #print all_delta_j_arr
                for r in range(len(curr_weights)):
                    for s in range(len(curr_weights[r])):
                        #print l_rate * curr_layer[s] * all_delta_j_arr[q][r]
                        curr_weights[r][s] = curr_weights[r][s] + l_rate * curr_layer[s] * all_delta_j_arr[q][r]

        #mse_lst.append(total_mse)
        print '###################################\n' + str(total_mse) + '\n##########################'

    #return (nn, mse_lst)

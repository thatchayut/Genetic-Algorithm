import numpy as np
import math

# ask for required value
def getInput(list_input_name):  
    while True:
        num_of_folds = input("Number of folds: ")
        if (num_of_folds.isnumeric() == False):
            print("WARNING : Invalid input must be numeric.")
        elif (int(num_of_folds) > len(list_input_name)):
            print("WARNING : Number of folds exceeds the size of the dataset.")
        elif (int(num_of_folds) <= 1):
            print("WARNING : Number of folds cannot lower than or equal to 1.")
        else:
            break
    num_of_folds = int(num_of_folds)

    # choose neural network architecture
    print("#### Select your neural network architecture ####")
    while True:
        num_of_layers = input("Number of hidden layers : ")
        if (num_of_layers.isnumeric() == False):
            print("WARNING : Invalid input must be numeric")
        elif (int(num_of_layers) < 1):
            print("WARNING : Number of layers must greater than or equal to 1.")
        else:
            break
    num_of_layers = int(num_of_layers)

    # choose number or nodes in each layer
    num_of_nodes_in_hidden_layer = []
    # separate input data in to k parts for cross-validation
    for i in range(0, num_of_layers):
        while True:
            num_of_nodes = input("Number of nodes in layer " + str(i + 1) + " : ")
            if (num_of_nodes.isnumeric() == False):
                print("WARNING : Invalid input must be numeric.")
            elif (int(num_of_nodes) < 1):
                print("WARNING : Number of nodes must greater than or equal to 1.")
            else:
                num_of_nodes = int(num_of_nodes)
                num_of_nodes_in_hidden_layer.append(num_of_nodes)
                break
    return num_of_folds, num_of_layers, num_of_nodes_in_hidden_layer

# split data l into n folds
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

# create individual by initialize weight
def createIndividual(num_of_hidden_layers, num_of_nodes_in_hidden_layer, num_of_input = 30, num_of_output = 1):
    # num_of_input = 30
    # num_of_output = 1
    # all layers that have weights  = hidden layer + output layer
    num_of_all_layer = num_of_hidden_layers + 1
    # create weight for hidden layer to hidden layer
    list_weight_hidden_hidden = []
    for layer_index in range(0, num_of_hidden_layers):
        list_weight_each_layer = []
        for node_index in range(0, num_of_nodes_in_hidden_layer[layer_index]):
            list_weight_each_node = []       
            # num_of_weight = weight from previous layer + weight bias
            # 1st element in weight_to_this_node is weight_bias
            # Input layer -> 1st hidden layer
            if (layer_index == 0):
                num_of_weight = num_of_input + 1
                weight_to_this_node = np.random.uniform(low = -1.0, high = 1.0, size = num_of_weight)
                list_weight_each_node.append(weight_to_this_node)
            # hidden layer -> hidden layer
            else:
                num_of_weight = num_of_nodes_in_hidden_layer[layer_index - 1] + 1
                weight_to_this_node = np.random.uniform(low = -1.0, high = 1.0, size = num_of_weight)
                list_weight_each_node.append(weight_to_this_node)
            list_weight_each_layer.extend(list_weight_each_node)
        list_weight_hidden_hidden.append(list_weight_each_layer)
    # craete weight for hidden to output
    list_weight_hidden_output = []
    for node_index in range(0, num_of_output):
        # number of weight from the last hidden layer to output node = number of node in the last hidden layer + weight bias
        num_of_weight = num_of_nodes_in_hidden_layer[num_of_hidden_layers - 1] + 1
        weight_to_this_node = np.random.uniform(low = -1.0, high = 1.0, size = num_of_weight)
        list_weight_hidden_output.append(weight_to_this_node)
    # combine all layers together
    list_all_weight = []
    list_all_weight.extend(list_weight_hidden_hidden)
    list_all_weight.append(list_weight_hidden_output)
    return list_all_weight

# create output from each node in network
def createY(num_of_hidden_layers, num_of_nodes_in_hidden_layer, num_of_input = 30, num_of_output = 1):
    list_Y_hidden = []
    for layer_index in range(0, num_of_hidden_layers):
        list_Y_each_layer = np.zeros(num_of_nodes_in_hidden_layer[layer_index])
        list_Y_hidden.append(list_Y_each_layer)
    list_Y_output = []
    for node_index in range(0, num_of_output):
        list_Y_each_layer = np.zeros(num_of_output)
        list_Y_output.append(list_Y_each_layer)
    list_all_Y = []
    list_all_Y.extend(list_Y_hidden)
    list_all_Y.extend(list_Y_output)
    return list_all_Y         

def scaling(list_input):
    max_value = max(list_input)
    min_value = min(list_input)
    list_result = []
    for element in list_input:
        result = (element - min_value) / (max_value - min_value)
        result = round(result, 7)
        list_result.append(result)
    return list_result

def sigmoid(x):
    # if (x < 0):
    result = (1 / (1 + math.exp(x)))
    result = round(result, 7)
    # elif (x >= 0):
        # result = (1 / (1 + math.exp(-x)))
        # result = round(result, 7)
    return result

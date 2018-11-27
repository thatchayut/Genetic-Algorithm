def getInput(list_input_name):
    # ask for required value
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
#!/usr/bin/python
import pandas as pd
import random
import process

def main():

    # get input and output data
    file_input = pd.read_csv("wdbc_input.csv")
    file_output = pd.read_csv("wdbc_output.csv")

    # get name of all sample
    list_input_name = []
    for element in file_input:
        list_input_name.append(element)
    
    # shuffle data to make each chunk eaech chunk does not depend on its order in a file
    random.shuffle(list_input_name)
    
    # ask for required value
    num_of_folds, num_of_layers, num_of_nodes_in_hidden_layer = process.getInput(list_input_name)

    print(num_of_nodes_in_hidden_layer)

if __name__ == '__main__':
    main()
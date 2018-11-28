#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import process
import math

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
    num_of_folds, num_of_hidden_layers, num_of_nodes_in_hidden_layer = process.getInput(list_input_name)

    # separate input in to k chunks
    chunk_size = math.ceil(len(list_input_name) / num_of_folds)
    chunk_sample = list(process.chunks(list_input_name, chunk_size))
    print(num_of_hidden_layers)
    print(num_of_nodes_in_hidden_layer)

    individual_1 = process.createIndividual(num_of_hidden_layers, num_of_nodes_in_hidden_layer)

    # class for each individuals

if __name__ == '__main__':
    main()
#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import process
import math
import copy

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
    num_of_chunks = len(chunk_sample)

    # individual_1 = process.createIndividual(num_of_hidden_layers, num_of_nodes_in_hidden_layer)

    # individual = {}
    # for i in range(0,5):
    #     key = i
    #     value = process.createIndividual(num_of_hidden_layers, num_of_nodes_in_hidden_layer)
    #     individual[key] = value
    # print(individual)

    # get data for each fold
    for test_sample_index in range(0, num_of_chunks):
        print("\n------------------------------------------ K : " + str(test_sample_index + 1) + " --------------------------------")
        test_sample = chunk_sample[test_sample_index]
        train_sample = []
        # select training data from all data by excluding testing data
        for train_sample_index in range(0, num_of_chunks):
            if (chunk_sample[train_sample_index] is not test_sample):
                train_sample.extend(chunk_sample[train_sample_index])
            # print(train_sample)

        # get data to train
        file_training_input = pd.read_csv("wdbc_input.csv", usecols = train_sample)
        file_training_output = pd.read_csv("wdbc_output.csv", usecols = train_sample)
        # print(len(file_training_input.columns))
     
        # create list of training data 
        num_of_samples = len(file_training_input.columns)
        print("num_of_samples = " + str(num_of_samples))
        list_training_input = []
        for column in range(0, num_of_samples):
            list_each_sample = []
            for element in file_training_input.iloc[:, column]:
                list_each_sample.append(element)
            list_training_input.append(list_each_sample)
        # print(list_training_input)
        
        list_training_output = []
        for column in range(0, num_of_samples):
            list_each_sample = []
            for element in file_training_output.iloc[:, column]:
                if (element == "M"):
                    list_each_sample.append(1)
                elif (element == "B"):
                    list_each_sample.append(0)
            list_training_output.append(list_each_sample)
    
        # scaling input to be in range (-1, 1)
        list_training_input_normalized = []
        for sample in list_training_input:
            result = process.scaling(sample)
            list_training_input_normalized.append(result)    
        print("list training normalized = " + str(list_training_input_normalized[0])) 
        # create all individuals in this population
        individuals = {}
        for i in range(0, num_of_samples):
            key = i
            value = process.createIndividual(num_of_hidden_layers, num_of_nodes_in_hidden_layer)
            individuals[key] = value
        
        # create a list to record output from each node
        list_all_Y = process.createY(num_of_hidden_layers, num_of_nodes_in_hidden_layer)

        # print(individuals[0])
        # print(len(individuals[0]))
        # print()
        # print(list_all_Y)
        # print(len(list_all_Y))
        # print("testttttttttttttt")
        # print(individuals[0][0][0][0])
        # find fitness function by forwarding
        # print("Architecture : " + str(individuals[0]))

        # ADJUST EPOCH FROM HERE!!!!!!!!!!!!!!!!

        # Forwarding
        # calculate fitness value of each individual
        list_fitness = np.zeros(num_of_samples)
        list_result = []
        for i in range(0, num_of_samples):
            # calcualte output for each node in hidden layers
            for layer_index in range(0, num_of_hidden_layers):
                for node_index in range(0, num_of_nodes_in_hidden_layer[layer_index]):
                    result = 0
                    # weight index is between 1 to len(individuals) because weight_index '0' is weight bias
                    num_of_weight = len(individuals[i][layer_index][node_index])
                    for weight_index in range(1, num_of_weight):
                        # for node in the 1st hidden layer
                        if (layer_index == 0):
                            # index of list_training_input_normalized must be the same index as the one for an individual
                            for element in list_training_input_normalized[0]:
                                result += (element * individuals[i][layer_index][node_index][weight_index])
                        # for other layers
                        else:
                            # y_this_node = sum(y_previous_node * weight_to_this_node)
                            for element in list_all_Y[layer_index - 1]:
                                result += (element * individuals[i][layer_index][node_index][weight_index])
                    # add bias to result (weight_index '0' is weight bias)
                    result += individuals[i][layer_index][node_index][0]
                    # apply activation function to result
                    result = process.sigmoid(result)
                    list_all_Y[layer_index][node_index] = result
            # print("result")
            # print(list_all_Y)
            # calculate output for output layer
            num_of_output = 1
            last_hidden_layer_index = len(individuals[i]) - 2
            # print(last_hidden_layer_index)
            last_layer_index = len(individuals[i]) - 1
            # print(last_layer_index)
            for output_index in range(0, num_of_output):
                # output = sum(y_previous_node * weight_to_this_node)
                result = 0
                for weight_index in  range(0, len(individuals[i][last_hidden_layer_index][output_index])):
                    for element in list_all_Y[len(list_all_Y) - 1]:
                        result += (element * individuals[i][last_hidden_layer_index][output_index][weight_index])
                # add bias to result (weight_index '0' is weight bias)
                result += individuals[i][last_layer_index][output_index][0]
                # print(individuals[0][last_layer_index][output_index][0])
                result = process.sigmoid(result)
                list_all_Y[last_layer_index][output_index] = result
            # print(list_all_Y)
            actual_output = list_all_Y[last_layer_index]
            desired_output = list_training_output[i]
            # print("Actual output : " + str(actual_output))
            # print("Desired output : " + str(desired_output))
            error = abs(desired_output[0] - actual_output[0])
            # print("Error : " + str(error))
            fitness_value = (1 / error)
            fitness_value = round(fitness_value, 7)
            # print("Fitness value : " + str(fitness_value))
            list_fitness[i] = fitness_value
            list_fitness = list(list_fitness)
        # print("list_fitness : " + str(list_fitness))
        max_fitness_index = list_fitness.index(max(list_fitness))
        # print(max_fitness_index)
        print()
        print("#### Result ####")
        print("Maximun finess value in this fold : " + str(list_fitness[max_fitness_index]))
        print("Optimal Structure in this fold : " + str(individuals[max_fitness_index]))

        # crossover
        # random number of individuals to be added to a mating pool
        num_individuals_in_mating = None
        while True:
            num_individuals_in_mating = random.randint(0, 10)
            if ((num_individuals_in_mating % 2) == 0):
                break
        print("Size of mating pool : " + str(num_individuals_in_mating))
        
        if (num_individuals_in_mating is not 0):
            # random individuals to be used as crossover elements
            list_individual_to_be_mated = []
            for i in range(0,num_individuals_in_mating):
                while True:
                    individual = random.randint(0, num_of_samples)
                    if individual not in list_individual_to_be_mated:
                        list_individual_to_be_mated.append(individual)
                        break
            print("Individuals in mating pool : " + str(list_individual_to_be_mated)) 

            # random crossing site for each individual in mating pool from 1 to number of layers in network
            list_crossing_site = []
            for i in range(0, num_individuals_in_mating):
                # random layer in each individuals to be a starting point for crossover (weight to first hidden layer and output layer are excluded)
                crossing_site = random.randint(1, num_of_hidden_layers)
                list_crossing_site.append(crossing_site)
            print("Crossing site for each individual : " + str(list_crossing_site))

            # paring individuals in mating pool
            list_paired_individuals = list(process.chunks(list_individual_to_be_mated, 2))
            list_paired_crossing_site = list(process.chunks(list_crossing_site, 2))
            print(list_paired_individuals)
            print(list_paired_crossing_site)

            # conducting crossover
            for pair_index in range(0, len(list_paired_individuals)):
                # index of individual in a pair
                index_first_individual = list_paired_individuals[pair_index][0]
                index_second_individual = list_paired_individuals[pair_index][1]
                # crossing site of each individual
                crossing_site_first_individual = list_paired_crossing_site[pair_index][0]
                crossing_site_second_individual = list_paired_crossing_site[pair_index][1]
                # create dummy used in swapping process
                temp_individuals_1 = copy.deepcopy(individuals[index_first_individual])
                temp_individuals_2 = copy.deepcopy(individuals[index_second_individual])
                # swap weight after crossing site to others
                for i in range(crossing_site_first_individual, len(temp_individuals_1)):
                    temp_individuals_2[i] = individuals[index_first_individual][i]         
                for i in range(crossing_site_second_individual, len(temp_individuals_2)):
                    temp_individuals_1[i] = individuals[index_second_individual][i]
                individuals[index_second_individual] = temp_individuals_2    
                individuals[index_first_individual] = temp_individuals_1
        
        # Mutation
        mutate_prob = 0.01
        num_of_mutation = math.ceil(num_of_hidden_layers * num_of_samples * mutate_prob)
        # random individuals to mutate
        list_individual_to_mutate = []
        for i in range(0, num_of_mutation):
            while True:
                individual = random.randint(0, num_of_samples)
                if individual not in list_individual_to_mutate:
                    list_individual_to_mutate.append(individual)
                    break
        # print("Individual to be mutated : " + str(list_individual_to_mutate))

        # Mutate individuals
        for i in range(0, len(list_individual_to_mutate)):
            individual_index = list_individual_to_mutate[i]
            # Random layer to mutate in each individual
            random_layer_index = random.randint(0, num_of_hidden_layers - 1)
            # Random new set of weight
            num_of_node_to_mutate = num_of_nodes_in_hidden_layer[random_layer_index]
            random_new_weight = np.random.uniform(low = -1.0, high = 1.0, size = num_of_node_to_mutate)
            # Mutate
            # individual_index = list_individual_to_mutate[i]
            individuals[individual_index][random_layer_index] = random_new_weight

if __name__ == '__main__':
    main()
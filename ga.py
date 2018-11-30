#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import process
import math
import copy
import time

def main():
    start_time = time.time()
    # get required value
    while True:
        mutate_prob = input("Mutatation probability : ")
        if (float(mutate_prob) > 1):
            print("WARNING : Probability can not greater than 1.")
        elif (float(mutate_prob) < 0):
            print("WARNING : Probability must be positive number")
        else:
            break
    while True:
        num_of_gen = input("Number of generations : ")
        if (num_of_gen.isnumeric() == False):
            print("WARNING : Probability must be numeric.")
        elif (int(num_of_gen) < 0):
            print("WARNING : Number of generation must be positive number")
        else:
            break

    output_file_name = input("Output file name : ")
    output_file = open(str(output_file_name) + ".txt", "w+")

    mutate_prob = float(mutate_prob)
    num_of_gen = int(num_of_gen)

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

    # get data for each fold
    for test_sample_index in range(0, num_of_chunks):
        print("\n------------------------------------------ K : " + str(test_sample_index + 1) + " --------------------------------")
        output_file.write("#################################### K : " + str(test_sample_index + 1) + " ####################################\n")
        test_sample = chunk_sample[test_sample_index]
        train_sample = []
        # select training data from all data by excluding testing data
        for train_sample_index in range(0, num_of_chunks):
            if (chunk_sample[train_sample_index] is not test_sample):
                train_sample.extend(chunk_sample[train_sample_index])

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

        # TRAINING
        # Iterate through generations
        for count_generation in range(0, num_of_gen):
            print(" #### Generation " + str(count_generation + 1) + " ####")
            # Forwarding
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
                        individual = random.randint(0, num_of_samples - 1)
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
                    if (list_paired_individuals[pair_index][0] != list_paired_individuals[pair_index][1]):
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
            # mutate_prob = 0.01
            num_of_mutation = math.ceil(num_of_samples * mutate_prob)
            print("num_of_mutation = " + str(num_of_mutation))
            # random individuals to mutate
            list_individual_to_mutate = []
            count = 0
            for i in range(0, num_of_mutation):
                while True:
                    individual = random.randint(0, num_of_samples - 1)
                    if (individual not in list_individual_to_mutate):
                        list_individual_to_mutate.append(individual)
                        count += 1
                        print(count)
                        break
            # print("Individual to be mutated : " + str(list_individual_to_mutate))

            # Mutate individuals
            for i in range(0, len(list_individual_to_mutate)):
                individual_index = list_individual_to_mutate[i]
                print("individual_index = " + str(individual_index))
                # Random layer to mutate in each individual
                random_layer_index = random.randint(0, num_of_hidden_layers)
                random_node_index = None
                if (random_layer_index == 0):
                    random_node_index = random.randint(0, num_of_nodes_in_hidden_layer[random_layer_index] - 1)
                else:
                    random_node_index = random.randint(0, num_of_nodes_in_hidden_layer[random_layer_index - 1] - 1)
                print("random_layer_index = " + str(random_layer_index))
                print("random node index = " + str(random_node_index))
                # layer 0 is weight connected to input layer
                if (random_layer_index == 0):
                    num_of_node_to_mutate = 30

                    random_new_weight = np.random.uniform(low = -1.0, high = 1.0, size = num_of_node_to_mutate)
                    print("BEFORE: " + str(individuals[individual_index][random_layer_index][random_node_index]))
                    individuals[individual_index][random_layer_index][random_node_index] = random_new_weight
                    print("AFTER : " + str(individuals[individual_index][random_layer_index][random_node_index]))
                else:
                    #change random index to fit with num_of_node_in_hidden_layer
                    random_layer_index  -= random_layer_index
                    # Random new set of weight
                    print(individuals[individual_index])
                    num_of_node_to_mutate = len(individuals[individual_index][random_layer_index][random_node_index])
                    # print("random_layer_index : " + str(random_layer_index))
                    # print("num_of_node_in_hidden_layer = " + str(num_of_nodes_in_hidden_layer))
                    # print("num of nodes : " + str(num_of_node_to_mutate))
                    random_new_weight = np.random.uniform(low = -1.0, high = 1.0, size = num_of_node_to_mutate)
                    # print("random_new_weight = " + str(random_new_weight))
                    # Mutate
                    # individual_index = list_individual_to_mutate[i]
                    print("BEFORE: " + str(individuals[individual_index][random_layer_index][random_node_index]))
                    individuals[individual_index][random_layer_index][random_node_index] = random_new_weight
                    print("AFTER : " + str(individuals[individual_index][random_layer_index][random_node_index]))

        # TESTING
        # preparing testing data
        # get data to test
        file_testing_input = pd.read_csv("wdbc_input.csv", usecols = test_sample)
        file_testing_output = pd.read_csv("wdbc_output.csv", usecols = test_sample)
        # create list testing data
        num_of_samples_to_test = len(file_testing_input.columns)
        print("num_of_samples_to_test = " + str(num_of_samples_to_test))
        list_testing_input = []
        for column in range(0, num_of_samples_to_test):
            list_each_sample = []
            for element in file_testing_input.iloc[:, column]:
                list_each_sample.append(element)
            list_testing_input.append(list_each_sample)

        list_testing_output = []
        for column in range(0, num_of_samples_to_test):
            list_each_sample = []
            for element in file_testing_output.iloc[:, column]:
                if (element == "M"):
                    list_each_sample.append(1)
                elif (element == "B"):
                    list_each_sample.append(0)
            list_testing_output.append(list_each_sample)

        # scaling input to be in range (-1, 1)
        list_testing_input_normalized = []
        for sample in list_testing_input:
            result = process.scaling(sample)
            list_testing_input_normalized.append(result)    
        print("list testing normalized = " + str(list_testing_input_normalized[0]))    

        # create a list to record output from each node
        list_all_Y_test = process.createY(num_of_hidden_layers, num_of_nodes_in_hidden_layer)

        # FORWARDING
        list_fitness = np.zeros(num_of_samples)
        list_result = []
        for i in range(0, num_of_samples_to_test):
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
            # calculate error
            error = abs(desired_output[0] - actual_output[0])
            # print("Error : " + str(error))
            # calculate fitness value
            fitness_value = (1 / error)
            fitness_value = round(fitness_value, 7)
            # print("Fitness value : " + str(fitness_value))
            list_fitness[i] = fitness_value
            list_fitness = list(list_fitness)

            # convert output to class
            if (actual_output < 0.5):
                converted_output = 0
                list_result.extend(str(converted_output))
            else:
                converted_output = 1
                list_result.extend(str(converted_output))
        max_fitness_index = list_fitness.index(max(list_fitness))
        # print(max_fitness_index)
        print("Actual output : " + str(list_result))
        print("Desired output : " + str(list_training_output))
        # print(list_result[0])
        # print(list_training_output[0][0])

        # Evaluation
        count_match = 0
        for i in range(0, len(list_result)):
            if(int(list_result[i]) == int(list_training_output[i][0])):
                count_match += 1
        accuracy = ((count_match / len(list_result)) * 100)
        print()
        print("#### Result ####")
        print("Accuracy : " + str(accuracy) + " %")
        print("Maximun fitness value in this fold : " + str(list_fitness[max_fitness_index]))
        print("Optimal Structure in this fold : " + str(individuals[max_fitness_index]))
        
        # change format of out put to be written to log file
        list_output_to_log = []
        for i in range(0, len(list_result)):
            list_output_to_log.append(list_training_output[i][0])
        # write output to log file
        output_file.write("Actual output : \n")
        output_file.write(str(list_result) + "\n")
        output_file.write("Desire output : " + "\n")
        output_file.write(str(list_output_to_log) + "\n")
        output_file.write("Accuracy : " + str(accuracy) + " %\n")
        output_file.write("Maximun fitness value of this fold : " + str(list_fitness[max_fitness_index]) + "\n")
        output_file.write("Optimal Structure in this fold : " + str(individuals[max_fitness_index]) + "\n")
    end_time = time.time()
    elapse_time_sec = end_time - start_time
    elapse_time_min = (elapse_time_sec / 60)
    elapse_time_min = round(elapse_time_min, 5)
    print("Total time elapse : " + str(elapse_time_min) + " minutes")
    output_file.write("\n Total elapse time : " + str(elapse_time_min) + " minutes")

if __name__ == '__main__':
    main()
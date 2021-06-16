import csv
from os import read
import numpy as np
from numpy.lib.function_base import interp
import random


def GetClusterDetails():
    nodes_number = input("Please insert the number of nodes in your cluster: \n")
    #print(f'You entered {nodes_number} and its type is {type(nodes_number)}')
    nodes_number = int(nodes_number)
    
    return nodes_number

def GetNumberOfFunctions():
    file = open("/home/giannos-g/Desktop/gavrielides_thesis/python_profiling/App_Info_Output_File_CSV.csv")
    reader = csv.reader(file)
    lines = len(list(reader))
    file.close()
    number_of_functions = lines - 1
    number_of_functions = int(number_of_functions)
    
    return number_of_functions

def GetWeightsArray():
    # Weights of each function
    # [Memory, Time, no_of_calls]
    mem_file = open("/home/giannos-g/Desktop/gavrielides_thesis/python_profiling/App_Info_Output_File_CSV.csv")
    table = []
    for line in mem_file:
        row = []
        for part in line.split():
            x = line.split(',')
            row.append(x[1])
            row.append(x[2])
            x2 = x[3].split('\n')
            row.append(x2[0])
        
        table.append(row)

    return_table = np.delete(table, 0, 0)               # Delete the first row
    mem_file.close()
    return_table = return_table.astype(float)
    
    return return_table

def GetNodeResources(nodes):
    # Each Nodes resources limit
    # [Memory, CPU]
    file = open ("/home/giannos-g/Desktop/gavrielides_thesis/cpu_mem_usage/Cluster_Details_CSV.csv")
    table = []
    for line in file:
        row = []
        for part in line.split(','):
            x = line.split(',')
            row.append(part)         # Energy column & CPU Column            

        table.append(row)

    for i in range(nodes-1):            # assumption: Every node has the same Resources
        table.append(row)
    
    return_table = np.delete(table,0,0)             # Delete the first row
    file.close()
    return_table = return_table.astype(float)
    
    return return_table

def initialize_x_array(nodes, functions):       # Columns = No.OfNodes
    table = []                                  # Rows = No.OfFunctions
    var = 0
    for r in range(int(functions)):
        row = []
        for c in range(int(nodes)):
            row.append(var)
        table.append(row)
    return table

def RandomlyFillTable(my_table, nodes, functions, nodes_resources, weights):
    choice_sequence = []
    
    for r in range(nodes):                      # number of nodes / columns of the table
        choice_sequence.append(r)
    print ("Choice sequence: \n", choice_sequence)
    
    # Let's fill the table
    for r in range(functions):
        random_node_choice = random.choice(choice_sequence)
        # Weights check
        if (nodes_resources[random_node_choice][1] > weights[r][0]):             #node resources[1] = memory available 
                                                                                 # weights[r][0] = memory needed for function r
                                                                                 # All nodes have the same available memory
            my_table[r][random_node_choice] = 1
            nodes_resources[random_node_choice][1] = nodes_resources[random_node_choice][1] - weights[r][0]      

    return my_table

def GetEnergyForEachFunction():
    predictions_table = []
    with open('/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/Predictions.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[0], part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    #predictions_table = predictions_table.astype(float)

    return predictions_table        # This is a 2D array

    # #file = open("")                 # Get energy from trained model - prediction
    # file = open("/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/Predictions.csv")
    # for line in file:
    #     reader = line.split()
    #     print(reader)
    #     energy = reader[0]
    # #print("ENERGY = ", energy)
    # energy_of_each_function = []
    # # Dummy Initialization
    # table = []                                  # Rows = No.OfNodes
    # var = 0                                     # Columns = 1
    # for r in range(1):
    #     row = []
    #     for c in range(int(functions)):
    #         row.append(var)
    #     table.append(row)

    # return table

def Manipulate_function_energy():
    predictions_table = []
    with open('/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/Predictions.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    predictions_table = predictions_table.astype(float)

    return predictions_table.T

def GetTotalEnergyMatrix(energy_array, x_array, nodes):     # Sum of energy of each node
    total_energy = np.matmul(energy_array, x_array)
    
    return total_energy

def GetTotalEnergy(energy_array, nodes):
    tot_energy = 0
    for r in range(int(nodes)):
        tot_energy += energy_array[0][r]

    return tot_energy

def main():
    Total_Energy = 0
    number_of_nodes = GetClusterDetails()               # m
    number_of_functions = GetNumberOfFunctions()        # n
    print("You have: ", number_of_functions, " functions to be destributed to, ", number_of_nodes, " nodes \n")
    weights = GetWeightsArray()
    print("Weights: Memory, Time, No.Calls \n", weights)
    #node_resources = GetNodeResources(number_of_nodes)
    #print ("Nodes Resources: \n", node_resources)
    #x_init = initialize_x_array(number_of_nodes, number_of_functions)
    #map_table = RandomlyFillTable(x_init,number_of_nodes, number_of_functions, node_resources, weights)
    #print ("Map array: \n", map_table)
    #function_energy = GetEnergyForEachFunction()
    #function_energy_only_energy_column = Manipulate_function_energy()
    #print ("This is the 1D Array: \n", function_energy_only_energy_column)
    #total_energy_matrix = GetTotalEnergyMatrix(function_energy_only_energy_column, x_init, number_of_nodes)
    #print("Total Energy Matrix: \n", total_energy_matrix)
    #total_energy = GetTotalEnergy(total_energy_matrix, number_of_nodes)
    #print("Total Energy = ", total_energy)
    for i in range (5):
        
        node_resources = GetNodeResources(number_of_nodes)
        x_init = initialize_x_array(number_of_nodes, number_of_functions)
        map_table = RandomlyFillTable(x_init,number_of_nodes, number_of_functions, node_resources, weights)
        function_energy = GetEnergyForEachFunction()
        function_energy_only_energy_column = Manipulate_function_energy()
        total_energy_matrix = GetTotalEnergyMatrix(function_energy_only_energy_column, x_init, number_of_nodes)
        total_energy = GetTotalEnergy(total_energy_matrix, number_of_nodes)
        print("Total Energy = ", total_energy)

        if (i == 0):
            Total_Energy = total_energy
            Final_Map_Array = map_table
            Final_Node_Resources = node_resources

        if (total_energy < Total_Energy):
            Final_Map_Array = map_table
            Total_Energy = total_energy

    print ("The best Array is: \n", Final_Map_Array)
    print ("The lowest energy is: \n" , Total_Energy)
    print ("The final Nodes Resources for the best case is: \n", Final_Node_Resources)

if __name__ == "__main__":
    main()

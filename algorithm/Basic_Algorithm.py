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
    # [Memory, CPU Utilization]
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
    #print ("Choice sequence: \n", choice_sequence)
    
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

def GetCommunications(functions):
    #Initialize the communications table
    communications_table = []                                  # Rows = No.OfFunctions
    var = 0                                                    # Columns = No.OfFunctions
    for r in range(int(functions)):
        row = []
        for c in range(int(functions)):
            row.append(var)
        communications_table.append(row)
    # Lets fill it 
    i = 0 
    functions_table = []
    read_communications_table = []
    source_functions_file = open('/home/giannos-g/Desktop/gavrielides_thesis/python_profiling/App_Info_Output_File_CSV.csv')
    communications_file = open('/home/giannos-g/Desktop/gavrielides_thesis/python_profiling/Calls_Details_CSV.csv')
    for line in source_functions_file:
        part = line.split(",")
        function_name = part[0]    
        functions_table.append(function_name)
    functions_table = np.delete(functions_table, 0, 0)      # delete first row
    
    rows = 0        # rows of the communications file
    for line in communications_file:
        row =[]
        part = line.split('\n')                 #each line
        # I need part[0] =>  has the 3 values 
        part2 = part[0].split(',')
        row.append(part2[0])
        row.append(part2[1])
        row.append(part2[2])
        read_communications_table.append(row)
        rows+=1
  
    read_communications_table = np.delete(read_communications_table, 0, 0)
    rows+=-1
    
    for i in range (0,functions):
        for j in range (0,rows-1):
            if (functions_table[i] == read_communications_table[j][0]):     # found the function
                communications_table[i][j+1] = int(read_communications_table[j][2])

    return communications_table 

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

def GetCostOfCommunications(communications, map_table,functions,nodes):
    total_cost_of_communication = 0
    for i in range (functions):
        for j in range (functions):
            if (communications[i][j] != 0):
                # Check where each one is executed:
                for jj in range (nodes):
                    if (map_table[i][jj] == map_table[j][jj] ):      # they are executed on the same node
                        #print ("They are executed on the same one")
                        total_cost_of_communication += communications[i][j] * 5     # Assumption = 5 Joule
                        break
                    else:
                        #print (("They are NOT executed on the same one"))
                        total_cost_of_communication += communications[i][j] * 20     # Assumption = 20 Joule
                        break

    return total_cost_of_communication

def GetTotalEnergy(energy_array, nodes, communication_cost):
    tot_energy = 0
    for r in range(int(nodes)):
        tot_energy += energy_array[0][r]

    tot_energy += communication_cost

    return tot_energy

def main():
    #iterations = input("Set the number of iterations \n")
    iterations = 50
    Total_Energy = 0
    number_of_nodes = GetClusterDetails()               # m
    number_of_functions = GetNumberOfFunctions()        # n
    print("You have: ", number_of_functions, " functions to be destributed to, ", number_of_nodes, " nodes \n")
    weights = GetWeightsArray()
    print("Weights: \n","Memory, Time, No.Calls \n", weights)
    # Lets start trying combinations
    for i in range (iterations):
        
        node_resources = GetNodeResources(number_of_nodes)
        x_init = initialize_x_array(number_of_nodes, number_of_functions)
        map_table = RandomlyFillTable(x_init,number_of_nodes, number_of_functions, node_resources, weights)
        function_energy = GetEnergyForEachFunction()
        function_energy_only_energy_column = Manipulate_function_energy()
        total_energy_matrix = GetTotalEnergyMatrix(function_energy_only_energy_column, x_init, number_of_nodes)
        communications = GetCommunications(number_of_functions)
        cost_of_communications = GetCostOfCommunications(communications, map_table, number_of_functions, number_of_nodes)
        total_energy = GetTotalEnergy(total_energy_matrix, number_of_nodes, cost_of_communications)
        
        print("Total Energy for iteration ", i+1," = ", total_energy, " (Joule)")

        if (i == 0):
            Total_Energy = total_energy
            Final_Map_Array = map_table
            Final_Node_Resources = node_resources

        if (total_energy < Total_Energy):
            Final_Map_Array = map_table
            Total_Energy = total_energy
    
    print ("Iterations: ", iterations)
    print ("The best Map Array is: \n", Final_Map_Array)
    print ("The lowest energy using the best Map Array is: \n" , Total_Energy, "(Joule)")
    #print ("The final Nodes Resources for the best case is: \n", Final_Node_Resources)

if __name__ == "__main__":
    main()

import csv
import numpy as np

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


""" def GetNodeResources():
    # Each Nodes resources limit
    # [Memory, CPU?????????????????]
    file = open ("")
    table = []
    for line in file:
        row = []
        for part in line.split(","):
            row.append(part[0])         # Energy column
            row.append(part[1])         # CPU Column
        
        table.append(row)

    return_table = np.delete(table,0,0)             # Delete the first row
    file.close()
    table = table.astype(float)
    return return_table
    #return 1D ????? array  """

def initialize_x_array(nodes, functions):
    table = []
    var = 0
    for r in range(int(functions)):
        row = []
        for c in range(int(nodes)):
            row.append(var)
        table.append(row)

    return table

def GetEnergyForEachFunction():
    file = open("")
    energy_of_each_function = []

    return energy_of_each_function

def GetEnergyOnEachNode():
    file = open("")                 # Get energy from trained model - prediction
    energy_on_nodes_table = []

    return energy_on_nodes_table


def GetTotalEnergy(node_array):
    total_energy =0

    for i in node_array:
        total_energy += node_array[i]

    return  total_energy


def main():
    number_of_nodes = GetClusterDetails()               # m
    number_of_functions = GetNumberOfFunctions()        # n
    print("The number of functions is:", number_of_functions, "\n")
    weights = GetWeightsArray()
    print("Weights: \n", weights)
    #node_resources = GetNodeResources()
    x_init = initialize_x_array(number_of_nodes, number_of_functions)
    function_energy = GetEnergyForEachFunction()
    node_energy_array = GetEnergyOnEachNode()
    total_energy = GetTotalEnergy(node_energy_array)
    

if __name__ == "__main__":
    main()

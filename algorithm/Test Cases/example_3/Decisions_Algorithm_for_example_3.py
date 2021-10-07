import csv
from os import read
import numpy as np
from numpy.lib.function_base import interp
import random


Expected_Total_Time = 0

def GetTimeThreshold():
    time_threshold = input("Please insert the time threshold (s) of your choice: \n")
    time_threshold = int(time_threshold)

    return time_threshold

def GetNanoDetails():
    nano_number = input("Please insert the number of Nano nodes in your cluster: \n")
    nano_number = int(nano_number)

    return nano_number

def GetJetsonDetails():
    jetson_number = input("Please insert the number of Xavier nodes in your cluster: \n")
    jetson_number = int(jetson_number)

    return jetson_number

def GetClusterDetails(nano, jetson):    # Enter the number of nodes of each chip
    # nano = GetNanoDetails()
    # jetson = GetJetsonDetails()
    # uno = GetUnoDetails()

    sumof_nodes_number = nano + jetson
    
    return sumof_nodes_number

def GetNumberOfFunctions():
    file = open("/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/python_profiling/App_Info_Output_File_CSV.csv")
    reader = csv.reader(file)
    lines = len(list(reader))
    file.close()
    number_of_functions = lines - 1
    number_of_functions = int(number_of_functions)
    
    return number_of_functions

def GetWeightsArray():
    # Weights of each function
    # [Memory, Time, no_of_calls]
    mem_file = open("/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/python_profiling/App_Info_Output_File_CSV.csv")
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

def GetNodeResources(nodes):        # Assumption: Every node has the same Resources
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
    source_functions_file = open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/python_profiling/App_Info_Output_File_CSV.csv')
    communications_file = open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/python_profiling/Calls_Details_CSV.csv')
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
        for j in range (0,rows):
            if (functions_table[i] == read_communications_table[j][0]):     # found the function
                for ii in range(0, functions):
                    if (functions_table[ii] == read_communications_table[j][1]):
                        communications_table[i][ii] = int(read_communications_table[j][2])
                """ print ("i = ", i)
                print ("j = ", j)
                #communications_table[i][j-1] = int(read_communications_table[j][2])
                print ("table = ", communications_table) """

    return communications_table 



def GetEnergyForEachFunction_on_Nano():   #NOT USED
    predictions_table = []
    with open('/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/Predictions.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[0], part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    #predictions_table = predictions_table.astype(float)

    return predictions_table        # This is a 2D array

def GetEnergyForEachFunction_on_Jetson():   #NOT USED
    predictions_table = []
    # OPEN THE CORRECT FILE
    with open('/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/Predictions.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[0], part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    #predictions_table = predictions_table.astype(float)

    return predictions_table        # This is a 2D array

def GetEnergyForEachFunction_on_Uno():      #NOT USED
    predictions_table = []
    # OPEN THE CORRECT FILE
    with open('/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/Predictions.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[0], part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    #predictions_table = predictions_table.astype(float)

    return predictions_table        # This is a 2D array



def Energy_Prediction_Table_on_Nano():
    predictions_table = []    
    # Prediction for Nano
    with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/energy_prediction_modeling/Energy_Predictions_on_nano.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    predictions_table = predictions_table.astype(float)
    

    return predictions_table.T

def Energy_Prediction_Table_on_Jetson():
    predictions_table = []    
    # Prediction for Jetson
    with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/energy_prediction_modeling/Jetson_xavier_nx_00/Predictions_jetson_xavier_nx_00.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    predictions_table = predictions_table.astype(float)

    return predictions_table.T

def GetTotalEnergyAndTimeMatrix(energy_array_nano,energy_array_jetson, map_table, number_of_nano,number_of_jetson, 
time_on_nano_table, time_on_jetson_table):     # Sum of energy of each node
    # Split the main map table into 2 others, one for each type of node
    functions = GetNumberOfFunctions()
    # Nano Map Table
    nano_map_table = []
    row = []
    for i in range(0,functions ):

        for j in range(0,number_of_nano):
            row.append(map_table[i][j])
        
        nano_map_table.append(row)
        row = []
    ###################################

    # Jetson Map Table
    jetson_map_table = []
    row = []
    for i in range(0,functions ):

        for j in range(number_of_nano,(number_of_nano + number_of_jetson)):
            row.append(map_table[i][j])
        
        jetson_map_table.append(row)
        row = []
    ###################################

    total_energy_on_each_nano_node = np.matmul(energy_array_nano, nano_map_table)
    total_energy_on_each_jetson_node = np.matmul(energy_array_jetson, jetson_map_table)
    
    total_time_on_nano = np.matmul(time_on_nano_table, nano_map_table)
    total_time_on_jetson = np.matmul(time_on_jetson_table, jetson_map_table)

    #print ("Nano Total time Table:\n", total_time_on_nano)
    #print ("Jetson Total time Table:\n", total_time_on_jetson)

    GetTotalTime(total_time_on_nano, total_time_on_jetson, number_of_nano, number_of_jetson)

    #print("Nano Map Table:\n", nano_map_table)
    #print("Jetson Map Table:\n", jetson_map_table)

    total_energy = np.append(total_energy_on_each_nano_node,total_energy_on_each_jetson_node)

    return total_energy

def GetCostOfCommunications(communications, map_table,functions,nodes): # Every node has the same cost 
    total_cost_of_communication = 0
    for i in range (functions):
        for j in range (functions):
            if (communications[i][j] != 0):
                # Check where each one is executed:
                for jj in range (nodes):
                    if (map_table[i][jj] == map_table[j][jj] ):      # they are executed on the same node
                        #print ("They are executed on the same one")
                        total_cost_of_communication += communications[i][j] * 2     # Assumption = 2 Joule
                        break
                    else:
                        #print (("They are NOT executed on the same one"))
                        total_cost_of_communication += communications[i][j] * 5     # Assumption = 5 Joule
                        break

    return total_cost_of_communication

def GetTotalEnergy(energy_array, nodes, communication_cost):
    tot_energy = 0
    for r in range(int(nodes)):
        tot_energy += energy_array[r]

    tot_energy += communication_cost
    return tot_energy

def GetTotalTimeOnNano():
    predictions_table = []    
    # Prediction for Nano
    with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/time_prediction_modeling/Nano/Time_Predictions_on_Nano.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    predictions_table = predictions_table.astype(float)

    return predictions_table.T

def GetTotalTimeOnJetson():
    predictions_table = []    
    # Prediction for Jetson
    with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/example_3/time_prediction_modeling/Jetson/Time_Predictions_on_Jetson.csv', 'r', newline='')as f:
        for line in f:
            part = line.split(',')
            prediction_row = [part[1]]
            predictions_table.append(prediction_row) #part[0] = name of functions

    predictions_table= np.delete(predictions_table, 0, 0)               # Delete the first row

    predictions_table = predictions_table.astype(float)

    return predictions_table.T

def GetTotalTime(time_on_nano_table, time_on_jetson_table,nano_nodes, jetson_nodes):
    Total_Time = 0
    global Expected_Total_Time      # To access the global Variable

    for r in range(int(nano_nodes)):
        Total_Time += time_on_nano_table[0][r] 

    for r in range(int(jetson_nodes)):
        Total_Time += time_on_jetson_table[0][r]

    Expected_Total_Time = Total_Time

    return Total_Time

def main():
    #iterations = input("Set the number of iterations \n")
    iterations = 10000
    Total_Energy = 0
    Best_Expected_Total_Time = 0
    Time_threshold = GetTimeThreshold()
    number_of_nano_nodes = GetNanoDetails()
    number_of_jetson_nodes = GetJetsonDetails()
    number_of_nodes = GetClusterDetails(number_of_nano_nodes, number_of_jetson_nodes)               # m
    number_of_functions = GetNumberOfFunctions()        # n
    print("You have: ", number_of_functions, " functions to be destributed to, ", number_of_nodes, " nodes \n")
    weights = GetWeightsArray()
    #print("Weights: \n","Memory, Time, No.Calls \n", weights)
    
    # Lets start trying combinations
    for i in range (iterations):
        
        node_resources = GetNodeResources(number_of_nodes)
        x_init = initialize_x_array(number_of_nodes, number_of_functions)
        map_table = RandomlyFillTable(x_init,number_of_nodes, number_of_functions, node_resources, weights)
        #function_energy_on_Nano = GetEnergyForEachFunction_on_Nano()
        function_energy_only_energy_column_nano = Energy_Prediction_Table_on_Nano()
        function_energy_only_energy_column_jetson = Energy_Prediction_Table_on_Jetson()
        # Calculate Time for this iteration
        time_on_nano_table = GetTotalTimeOnNano()
        time_on_jetson_table = GetTotalTimeOnJetson()

        total_energy_matrix = GetTotalEnergyAndTimeMatrix(function_energy_only_energy_column_nano,
                                                    function_energy_only_energy_column_jetson,
                                                    map_table, number_of_nano_nodes, 
                                                    number_of_jetson_nodes, time_on_nano_table,
                                                    time_on_jetson_table)
        communications = GetCommunications(number_of_functions)
        cost_of_communications = GetCostOfCommunications(communications, map_table, number_of_functions, number_of_nodes)
        total_energy = GetTotalEnergy(total_energy_matrix, number_of_nodes, cost_of_communications)
        
        #print("Total Time for iteration ", i+1," = ", Expected_Total_Time, " (s)")
        #print("Total Energy for iteration ", i+1," = ", total_energy, " (Joule)")
        
        
        if ((Expected_Total_Time <= Time_threshold) or (i==0)):
            if (i == 0):
                Total_Energy = total_energy
                Final_Map_Array = map_table
                Final_Node_Resources = node_resources
                Best_Expected_Total_Time = Expected_Total_Time 

            if (total_energy < Total_Energy):
                Final_Map_Array = map_table
                Total_Energy = total_energy
                Best_Expected_Total_Time = Expected_Total_Time 
    
    #print ("Iterations: ", iterations)
    if (Best_Expected_Total_Time > Time_threshold):
        print ("There is no implementation with this low time threshold")
    else:
        print ("The best Map Array is: \n", Final_Map_Array)
        print ("The lowest energy using the best Map Array is: \n" , Total_Energy, "(Joule)")
        #print ("The final Nodes Resources for the best case is: \n", Final_Node_Resources)
        print ("The time using the best Map Array is: \n" , Best_Expected_Total_Time, "(s)")

        with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test Cases/Algorithm Results.csv', 'a', newline='')as f:
            thewriter=csv.writer(f)
            thewriter.writerow([number_of_functions, number_of_nano_nodes, number_of_jetson_nodes, Time_threshold,
            Best_Expected_Total_Time, Total_Energy, Final_Map_Array ])

            print ('Saved: ', number_of_functions, number_of_nano_nodes, number_of_jetson_nodes, Time_threshold,
            Best_Expected_Total_Time, Total_Energy, Final_Map_Array)

if __name__ == "__main__":
    main()

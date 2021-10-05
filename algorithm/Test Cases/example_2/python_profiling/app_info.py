import csv
import os
from prettytable import PrettyTable
filename = "example_2.py"


# A single node of a singly linked list
class Node:
  # Constructor
  def __init__(self, funct_name, memory, time, ncalls, next_node=None): 
    self.name = funct_name
    self.memory = memory
    self.time = time
    self.ncalls = ncalls
    self.next_node = next_node

  # Methods of Nodes
  def get_name(self):                   #returns the stored name
        return self.name

  def get_memory(self):                 #returns the stored memory
        return self.memory
        
  def get_time(self):                   #returns the stored time
        return self.time

  def get_ncalls(self):                 #returns the stored ncalls
        return self.ncalls

  def get_next(self):                   #returns the next node
        return self.next_node

  def set_next(self, new_next):         #resets the pointer to a new node
        self.next_node = new_next

# Head Node
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    #Methods of the List
    #Insert method -  inserts a new node into the list
    def insert(self, funct_name, memory, time, ncalls):
        new_node = Node(funct_name, memory, time, ncalls)
        new_node.set_next(self.head)
        self.head = new_node

    #Size method - returns the size of the list
    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count

    #Print method for the linked list
    def printLL(self):
        current = self.head
        while(current):
            print(current.name, current.time, current.memory, current.ncalls)
            current = current.next_node

    # Search method - searches list for a node containing 
    # the requested data and returns that node if found, otherwise raises an error
    def search(self, funct_name):
        current = self.head
        found = False
        while current and found is False:
            if current.get_name() == funct_name:
                found = True
            else:
                current = current.get_next()
        if current is None:
            raise ValueError("Function with this name could not be found in the list")
        return current

    # Delete method searches list for a node containing 
    # the requested data and removes it from list if found, otherwise raises an error
    def delete(self, funct_name):
        current = self.head
        previous = None
        found = False
        while current and found is False:
            if current.get_name() == funct_name:
                found = True
            else:
                previous = current
                current = current.get_next()
        if current is None:
            raise ValueError("Function with this name could not be found in the list")
        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())

# Search txt files to insert the new node
def search_and_create(mem_profiler, tm_profiler, LL):

 # For the Memory Profiler - I may not use it
 for line in mem_profiler:                  # Split line by line
    for part in line.split():               # Split the string to parts in a list 
           if "def" in part:                # Check if we are at the definition line which always starts with "def"
                x = line.split("def", 1)    # Split it to 2 parts, we need the last one x[1]
                text2 = x[1]
                x2 = text2.split (" ", 1)   # Split to 2 parts, we need the last one x2[1]
                text3 = x2[1]
                x3 = text3.split(":")       # Delete the ":"                          
                text4 = x3[0]               # Delete the ()
                x4 = text4.split("(",1)     # We have the name of the function in x3[0]
                
                # Find the line that the function is def
                n = line.split()            # n[0] holds the line that the function is def 

                # Method to get the Memory of each line
                memory = 0
                converted_num = int(n[0])       # turn str to int
                converted_num += 1 
                a = mem_profiler.readline().split()
                this = True
                while this: 
                    if not a:
                        this = False
                    else:
                        if (int(a[0]) == converted_num):
                            memory = memory + float(a[1])       # memory has the total memory used of each function
                    a = mem_profiler.readline().split()
                    converted_num = converted_num +1           
                LL.insert(x4[0], memory, 0, 0)             # Create the Nodes only by the funct name and memory and 
                                                           # initialize the other vars to 0
                
# For the Time Profiler
 for line in tm_profiler:                   # Split line by line
     for part in line.split():              # Split the string to parts in a list 
         if filename in part:  
                x = line.split(":")         # Split to 2 parts compare to :" we need the second one x[1]
                text2 = x[1]
                x2 = text2.split ("(")      # Split to 2 parts compare to "(" we need the second one x2[1]
                text3 = x2[1]
                x3 = text3.split(")")       # We have the name of the function in x3[0]
               
                # Find the ncalls and tottime
                y = line.split()                # y[0] holds the ncalls and y[1] holds the tottime
              
                # Search the Linked List to find the function and insert the new vars
                LL.search(x3[0]).ncalls = y[0]
                LL.search(x3[0]).time = y[1]
                
def main():
    # Creation of Linked List 
    LL = LinkedList()

    # Open the txt files
    ##################os.system("python -m memory_profiler example.py")
    memory_profiler = open(r"./Memory_Profiling.txt", "r")
    time_profiler = open(r"./Time_Profiling.txt", "r")
    
    # Call search_and_create to create the list 
    search_and_create(memory_profiler, time_profiler, LL)

    # Close the txt files
    memory_profiler.close()
    time_profiler.close()

    #Print Output
    print ("Filename:", filename)
    t = PrettyTable(['Name', 'Memory', 'Time', 'nCalls'])
    current = LL.head
    while(current):
        t.add_row([current.name, current.memory, current.time, current.ncalls])
        current = current.next_node
    print(t)

    current = LL.head
    with open('./App_Info_Output_File_CSV.csv', 'w', newline='')as f:
        thewriter=csv.writer(f)
        thewriter.writerow(['Name', 'Memory', 'Time', 'nCalls'])
        while (current):
            thewriter.writerow([current.name, current.memory, current.time, current.ncalls])
            current = current.next_node

if __name__ == '__main__':
    main()




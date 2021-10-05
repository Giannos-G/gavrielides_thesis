import csv

class Node:
    #Constructor 
    def __init__(self, source_name, called_name, time, next_node=None): 
        self.src_name = source_name 
        self.cld_name = called_name 
        self.time = time
        self.next_node = next_node

    #Methods of Node
    def get_src_name(self):
        return self.src_name
    
    def get_cld_name(self):
        return self.cld_name
    
    def get_time(self):
        return self.time
    
    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next
    
# Head Node
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
    
    #Methods of the List
    #Insert method -  inserts a new node into the list
    def insert(self, source_name, called_name, time):
        new_node = Node(source_name, called_name, time)
        new_node.set_next(self.head)
        self.head = new_node


def search_calls_details(calls_det, LL):
    x = calls_det.readline().split()
    while (x[0]!= "edgedef>"):
        x = calls_det.readline().split()                                            # Line of edgedef<
    for line in calls_det:
            y = line.split(",")
            LL.insert(y[0], y[1], y[2])                                             #Create the new Node
            
            print ("Function -->" , y[0], " calls -->", y[1], y[2], "time(s)")      # That's what we want

def main():
    calls_details = open (r"/home/giannos/Desktop/giannos_thesis/python_profiling/pycallgraph.txt", "r")

    # Create the Linked LIst
    LL = LinkedList()
    
    #Call the function
    search_calls_details(calls_details, LL)

    calls_details.close()

    #Print the output
    current = LL.head
    with open('/home/giannos/Desktop/giannos_thesis/python_profiling/Calls_Details_CSV.csv', 'w', newline='')as f:
        thewriter=csv.writer(f)
        thewriter.writerow(['Source function', 'Function called', 'Time(s)'])
        while (current):
            thewriter.writerow([current.src_name, current.cld_name, current.time])
            current = current.next_node

if __name__ == '__main__':
    main()
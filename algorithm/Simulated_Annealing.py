import random

def RandomlyFillTable(my_table):
    choice_sequence = []
    for r in range(3):      # number of nodes / columns of the table
        choice_sequence.append(r)
    print ("Choice sequence: \n", choice_sequence)
    for r in range(5):
        random_node_choice = random.choice(choice_sequence)
        my_table[r][random_node_choice] = 1
    return my_table

def initialize_x_array():                       # Columns = No.OfNodes
    table = []                                  # Rows = No.OfFunctions
    var = 0
    for r in range(int(5)):
        row = []
        for c in range(int(3)):
            row.append(var)
        table.append(row)

    return table


def main():
    print("This is the random table: \n", RandomlyFillTable(initialize_x_array()))

if __name__ == "__main__":
    main()
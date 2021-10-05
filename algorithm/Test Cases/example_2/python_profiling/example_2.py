# Dummy Code containing multiple functions

#import cProfile

#@profile
def create_array():
  arr=[]
  for i in range(0,400000):
    arr.append(i)
  initialize_array(arr)


#@profile
def print_statement():
  print('Array created successfully')

#@profile
def initialize_array(my_array = [], *arr):
  for x in range(0,4000000):
    my_array.append(0)
  initialize_to_1_array (my_array)

#@profile
def initialize_to_1_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(1)
  print('Array initialized to 1 successfully')
  initialize_to_2_array (my_array_1)


#@profile
def initialize_to_2_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(2)
  print('Array initialized to 2 successfully')
  initialize_to_3_array (my_array_1)

#@profile
def initialize_to_3_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(3)
  print('Array initialized to 3 successfully')
  initialize_to_4_array (my_array_1)

#@profile
def initialize_to_4_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(4)
  print('Array initialized to 4 successfully')
  initialize_to_5_array (my_array_1)

#@profile
def initialize_to_5_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(5)
  print('Array initialized to 5 successfully')
  initialize_to_6_array (my_array_1)

#@profile
def initialize_to_6_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(6)
  print('Array initialized to 6 successfully')

#@profile
def main():
  create_array()
  print_statement()

if __name__ == '__main__':
    main()
    #cProfile.run('main()')
    
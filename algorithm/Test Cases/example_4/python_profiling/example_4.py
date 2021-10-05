# Dummy Code containing multiple functions

#import cProfile
import numpy as np

#@profile
def create_array():
  arr=[]
  for i in range(0,400000):
    arr.append(i)
  initialize_array(arr)

#@profile
def create_two_dimensions_array():
  arr_1=[]
  arr = []
  for i in range(0,100):

    for j in range (0,100):
      arr_1.append(j)

    arr.append(arr_1)
    arr_1=[]
  #print (arr)
  return arr

#@profile
def multiply_arrays_1():
  array1 = create_two_dimensions_array()
  array2 = create_two_dimensions_array()
  result = np.dot(array1, array2)
  print ("Arrays have been multiplied")

#@profile
def multiply_arrays_2():
  array1 = create_two_dimensions_array()
  array2 = create_two_dimensions_array()
  result = np.dot(array1, array2)
  print ("Arrays have been multiplied")

#@profile
def multiply_arrays_3():
  array1 = create_two_dimensions_array()
  array2 = create_two_dimensions_array()
  result = np.dot(array1, array2)
  print ("Arrays have been multiplied")

#@profile
def multiply_arrays_4():
  array1 = create_two_dimensions_array()
  array2 = create_two_dimensions_array()
  result = np.dot(array1, array2)
  print ("Arrays have been multiplied")

#@profile
def multiply_arrays_5():
  array1 = create_two_dimensions_array()
  array2 = create_two_dimensions_array()
  result = np.dot(array1, array2)
  print ("Arrays have been multiplied")

#@profile
def multiply_arrays_6():
  array1 = create_two_dimensions_array()
  array2 = create_two_dimensions_array()
  result = np.dot(array1, array2)
  print ("Arrays have been multiplied")


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
  initialize_to_7_array (my_array_1)

#@profile
def initialize_to_7_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(5)
  print('Array initialized to 7 successfully')
  initialize_to_8_array (my_array_1)

#@profile
def initialize_to_8_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(5)
  print('Array initialized to 8 successfully')
  initialize_to_9_array (my_array_1)

#@profile
def initialize_to_9_array(my_array_1 = [], *my_array):
  for x in range(0,4000000):
    my_array_1.append(5)
  print('Array initialized to 9 successfully')


#@profile
def main():
  create_array()
  print_statement()
  multiply_arrays_1()
  multiply_arrays_2()
  multiply_arrays_3()
  multiply_arrays_4()
  multiply_arrays_5()
  multiply_arrays_6()

if __name__ == '__main__':
    main()
    #cProfile.run('main()')
    
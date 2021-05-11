# python measure_time_energy.py "python example.py" 2
# Output:
# "output of the example file"
# time = ...
# ... (Joule)

import os

print (os.listdir())
x = os.listdir()
for i in range(1,4,1): 
    os.system("python3 " + x[i] )

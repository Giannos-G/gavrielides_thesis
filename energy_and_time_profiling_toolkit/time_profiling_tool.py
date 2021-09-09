import csv
import time
import os

path = "/home/giannos-g/Desktop/energy_profiling_toolkit/python_dataset/py_scripts"
x = os.listdir(path)

for i in x:
    if (i == 'energy_on_nano.py'):
        print ("Energy_On_Nano_Spotted and should not be measured")
    else:
        print ("-------------------- python script: "+ i +" --------------------")
        start_time = time.time()
        os.system('python3 ' + path +'/'+i)
        execution_time = time.time() - start_time
        print("--- %s seconds ---" % (time.time() - start_time))
        with open('/home/giannos-g/Desktop/energy_profiling_toolkit/Time_Profiling.csv', 'a')as f:
            thewriter=csv.writer(f)
            thewriter.writerow([i, execution_time])
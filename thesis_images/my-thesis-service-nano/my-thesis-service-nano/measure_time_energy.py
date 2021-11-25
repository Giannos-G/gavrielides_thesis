import sys
import os
#from profiller import profiller
#from application import application

import signal
import subprocess
import time

import collections
import itertools
import sys
import os.path

from energy_parser import *

# we do not need clang parser for this operation
#from parser import parser
#from transformer import transformer
import configparser
#from application.code_block import Code_block

if (len(sys.argv) < 4):
        print ("Usage: python measure_time_energy.py <application> <iterations>")
        exit()

print ("===========================================================================================")
#filename = sys.argv[2]
#x = filename.split()
#filename = x[1]
pro = subprocess.Popen("./get-power.sh")

#execute program

# Debugging......
print ("sys.arv[0] = ", sys.argv[0])
print ("sys.arv[1] = ", sys.argv[1])
print ("sys.arv[2] = ", sys.argv[2])
print ("sys.arv[3] = ", sys.argv[3])

time.sleep(0.5)
os.system("echo \"Start \" >> module-power-input.txt")
start = time.time()
#strs = sys.argv[2]
strs = sys.argv[1]+" "+sys.argv[2]
print (strs)
####### itan 2 to ekana 3
for i in range(1,int(sys.argv[3])):
	os.system(strs)
end = time.time()
os.system("echo \"End \" >> module-power-input.txt")

pro.kill()

print ("time = "+str(end - start)+" secs")
energy_results()

#with open('/home/gavrielides/python_dataset/py_scripts/Energy_Profiling_on_nano.csv', 'a')as f:
#	thewriter=csv.writer(f)
#	thewriter.writerow([filename, energy_results()])


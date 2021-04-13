Here we measure the time and the peak memory of a random python script. The profiler writes the results in the *csv* file so that we can use it later when we will analyze the energy each script produces to run in a raspberry pi. The goal is to make a big amount of data (with random python scripts) and then use them to predict the energy a python sciprt will produce in a rpi. 

### Sample:
- example.py
### Profiler:
- time_peak_mem_profiler.py 
### Output file:
- Energy_Profiling_Details_CSV.csv
### Modifier:
- clone_and_modify.py
- This script copies a file (bench_trees.py in this example) and modifies specific lines of the copied (or clone) file. This is done to help us accelerate the process of producing python scripts to be used in our dataset.   

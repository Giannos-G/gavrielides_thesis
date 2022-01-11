## Directory Description

Here we measure the time and the peak memory of a random python script. The profiler writes the results in a *csv* file so that we can use it later when we will analyze the energy each script produces to run in a device. The goal is to make a big amount of data *-dataset-* with random python scripts from **scikit-learn library** and then use them to predict the energy a python sciprt will produce in a specific edge device [^1]. 

### Sample:
- example.py
### Profiler:
- time_peak_mem_profiler.py 
### Output file:
- Energy_Profiling_Details_time_memoryCSV.csv
### Modifier:
- clone_and_modify.py
  - This script copies a file (bench_trees.py in this example *-see the code-*) and modifies specific lines of the copied (or cloned) file. This is done to help us accelerate the process of producing python scripts to be used in our dataset. Using this tool we can produce multiple similar python scripts automatically.  

*Reminder:*

![Datasets](https://user-images.githubusercontent.com/77551993/148947492-3d9177ad-4610-4fb9-820b-7e0c9d093796.png)

[^1]: Please ignore: *./New_Energy_Profiling_Details_CSV.csv ./time_profiling_tool.py ./energy_on_nano.py *

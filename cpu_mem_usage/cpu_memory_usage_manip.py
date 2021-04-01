import csv

def search_and_get_total_memory (memory):
    line_to_read = [1]
    for position, line in enumerate (memory):
        if position in line_to_read:
            x = line.split() 
    return x[6]


def search_and_get_total_cpu (usage):
    cpu_perc_usage = 0.0
    for i,line in enumerate(usage):
        if i >= 7:
                x = line.split()
                x2 = float(x[8].replace(',', '.'))
                cpu_perc_usage = cpu_perc_usage + x2
    return cpu_perc_usage

def main():
    cpu_usg = open(r"/home/giannos/Desktop/giannos_thesis/cpu_mem_cluster/cpu_usage.txt", "r")
    mem_usg = open(r"/home/giannos/Desktop/giannos_thesis/cpu_mem_cluster/memory_usage.txt", "r")        # It is given in KB
    total_cpu = search_and_get_total_cpu(cpu_usg)
    available_memory = search_and_get_total_memory(mem_usg)
    print ("Total CPU usage is:", total_cpu, "%")
    print ("Total Memory available:", available_memory, "KB")
    with open('/home/giannos/Desktop/giannos_thesis/cpu_mem_cluster/Cluster_Details_CSV.csv', 'w', newline='')as f:
        thewriter=csv.writer(f)
        thewriter.writerow(['Total CPU usage (%)', 'Total Memory Available (KB)'])
        thewriter.writerow([total_cpu, available_memory])

if __name__ == '__main__':
    main()

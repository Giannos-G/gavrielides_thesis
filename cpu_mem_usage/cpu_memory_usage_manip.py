
def search_and_get_total_memory (memory):
    line_to_read = [1]
    for position, line in enumerate (memory):
        if position in line_to_read:
            x = line.split()
    
    return x[6]


def search_and_get_total_cpu (usage):
    cpu_perc_usage = 0.0
    for readline in usage:
        x = readline.split()
        if x[2] != "%CPU":
            cpu_perc_usage = cpu_perc_usage + float(x[2])
    return cpu_perc_usage

def main():
    cpu_usg = open(r"/home/giannos/Desktop/CPU_and_RAM_usage/cpu_usage.txt", "r")
    mem_usg = open(r"/home/giannos/Desktop/CPU_and_RAM_usage/memory_usage.txt", "r")        # It is given in KB
    print ("Total CPU usage is:", search_and_get_total_cpu(cpu_usg), "%")
    print ("Total Memory available:", search_and_get_total_memory(mem_usg), "KB")

if __name__ == '__main__':
    main()

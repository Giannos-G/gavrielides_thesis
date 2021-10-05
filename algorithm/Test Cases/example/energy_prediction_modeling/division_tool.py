# This tool is used to divide the energy column of the dataset in 2
# because the metrics where taken for each script 2 times

import csv


def divide_and_create(source, new_file):
    #with open ("/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/my_dataset.csv", 'r') as f:
    #with open ("/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/my_new_dataset.csv",'w') as f:
    reader = csv.reader(source)
    writer = csv.writer(new_file)
    writer.writerow()

    retu



def main():
    source = open ("/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/my_dataset.csv", "r")
    new_file = open ("/home/giannos-g/Desktop/gavrielides_thesis/energy_prediction_modeling/my_new_dataset.csv", "w+")
    divide_and_create(source, new_file)


if __name__ == '__main__':
    main()
from math import log
from numpy.core.fromnumeric import size
from numpy.lib.utils import info
import pandas as pd 
import csv
import numpy as np

from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor

def main():
    #--------------Uploading dataset--------------
    data = pd.read_csv("/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test_Case/time_prediction_modeling/Nano/Time_profiling_Nano_details.csv")
    data.drop("PyScript", inplace=True, axis=1) # Delete 1st column => filenames of samples

    #------------------Remove the outliers------------------
    Q1 = data.quantile(0.12)
    Q3 = data.quantile(0.88)
    IQR = Q3 - Q1
    data_out = data[~((data < (Q1-1.5*IQR)) | (data > (Q3 +1.5*IQR))).any(axis=1)]

    #------------------Set data equals to new_data------------------
    data = data_out.copy()

    #--------------Construct the target column--------------
    target_column = data['Time on Nano(s)']

    #--------------Construct the data column--------------
    data.drop("Time on Nano(s)", inplace=True, axis=1)
    #===============================================================================
    #--------------Linear Regression-------------- 
    lr = LinearRegression()
    lr.fit(data, target_column)

    #--------------Predict data--------------
    predict_table = []
    info_table = []
    with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test_Case/python_profiling/App_Info_Output_File_CSV.csv', 'r', newline='')as f:
        #thereader=csv.reader(f)
        for line in f:
            part = line.split(',')
            prediction_row = [part[1], part[2]]
            predict_table.append(prediction_row) #part[0] = name of functions
            info_row = [part[0], part[1], part[2], part[3]]
            info_table.append(info_row)

    predict_table= np.delete(predict_table, 0, 0)               # Delete the first row
    info_table = np.delete(info_table, 0, 0)

    predict_table = predict_table.astype(float)

    y_pred_lr_test = lr.predict(predict_table)
    y_pred_lr_test = abs(y_pred_lr_test)

    # Get number of functions
    file = open("/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test_Case/python_profiling/App_Info_Output_File_CSV.csv")
    reader = csv.reader(file)
    lines = len(list(reader))
    file.close()
    number_of_functions = lines - 1
    number_of_functions = int(number_of_functions)

    for i in range(number_of_functions):
        print("My time prediction for function--> ", info_table[i][0], "<-- on Nano is: \n", y_pred_lr_test[i],"(s)")

    with open('/home/giannos-g/Desktop/gavrielides_thesis/algorithm/Test_Case/time_prediction_modeling/Nano/Time_Predictions_on_Nano.csv', 'w', newline='')as f:
        thewriter=csv.writer(f)
        thewriter.writerow(['Function Name', 'Time on Nano Prediction (s)'])
        for i in range(number_of_functions):
            print(info_table[i][0])
            thewriter.writerow([info_table[i][0], y_pred_lr_test[i]])


if __name__ == "__main__":
    main()
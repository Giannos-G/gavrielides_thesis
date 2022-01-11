## Directory Description

### NVIDIA Jetson Nano

- ./my_dataset.csv --> The dataset we created by running all of the dataset entries and capturing their run-time and energy consumption.
- ./energy_prediction_tool.py --> The tool we created in order to find the best ML model *(the model with the least MSE)*
- ./working_prediction_tool.py --> After deciding the best-fit ML model here we calculate the predictions
- ./Predictions.csv --> The output of ./working_prediction_tool.py *(our predictions)* [^1]

### NVIDIA Jetson Xavier

The same procedure has been followed in ./Jetson_xavier_nx_00

We now have *(considering time predictions have already been calculated)*:

![Copy of Datasets](https://user-images.githubusercontent.com/77551993/148943549-43b63133-e183-4398-8771-787a6439e6bd.png)


[^1]: Please ignore: *./Energy_Prediction.csv ./division_tool.py ./my_new_dataset.csv

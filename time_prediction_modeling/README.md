## Directory Description

### NVIDIA Jetson Nano 
*(./Nano)*

- ./Time_profiling_Nano_details.csv --> The dataset we created by running all of the dataset entries and capturing their run-time and energy consumption.
- ./energy_prediction_tool.py --> The tool we created in order to find the best ML model *(the model with the least MSE)*
- ./working_prediction_tool.py --> After deciding the best-fit ML model here we calculate the predictions
- ./Time_Predictions_on_Nano.csv --> The output of ./working_prediction_tool.py *(our predictions)*

### NVIDIA Jetson Xavier

The same procedure has been followed in ./Jetson

We now have *(considering energy predictions have already been calculated)*:

![Copy of Datasets](https://user-images.githubusercontent.com/77551993/148943549-43b63133-e183-4398-8771-787a6439e6bd.png)

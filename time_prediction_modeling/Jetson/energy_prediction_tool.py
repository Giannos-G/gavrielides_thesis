from math import log
from numpy.core.fromnumeric import size
from numpy.lib import math
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures


#--------------Uploading dataset--------------
data = pd.read_csv("/home/giannos-g/Desktop/gavrielides_thesis/time_prediction_modeling/Jetson/Time_profiling_Jetson_details.csv")
data.drop("PyScript", inplace=True, axis=1) # Delete 1st column => filenames of samples

#--------------Correlation Matrix of the original set--------------
cor_matrix = data.corr()
sn.heatmap(cor_matrix, annot = True)
plt.title('Correlation Matrix of the Original Dataset')
plt.show() 

#------------------Show Original Time Vs Energy------------------
plt.scatter(data['Time on local(s)'], data['Time on Jetson(s)'])
plt.title('Time on Jetson(s) = f(Time on local(s))')
plt.xlabel('Time on local(s)')
plt.ylabel('Time on Jetson(s)')
plt.show()

#------------------Show Time Vs Energy with limits------------------
plt.scatter(data['Time on local(s)'], data['Time on Jetson(s)'])
plt.xlim(0, 25)
plt.ylim(0, 100)
plt.title('Time on Jetson(s) = f(Time on local(s)) -with limits-')
plt.xlabel('Time on local(s)')
plt.ylabel('Time on Jetson(s)')
plt.show()

#------------------Show Original Memory Vs Energy------------------
plt.scatter(data['Memory(MB)'], data['Time on Jetson(s)']) 
plt.title('Time on Jetson(s) = f(Memory)')
plt.xlabel('Memory (MB)')
plt.ylabel('Time on Jetson(s)')
plt.show()

#------------------Show Memory Vs Energy with limits------------------
plt.scatter(data['Memory(MB)'], data['Time on Jetson(s)'])
plt.xlim(0, 500)
plt.ylim(0, 1000) 
plt.title('Time on Jetson(s) = f(Memory) -with limits-')
plt.xlabel('Memory (MB)')
plt.ylabel('Time on Jetson(s)')
plt.show()

#------------------Boxplot of the data------------------
data.boxplot(['Memory(MB)', 'Time on local(s)', 'Time on Jetson(s)'])
plt.ylim(0, 350)
plt.title('Boxplot of the original data')
plt.show()

#------------------Remove the outliers------------------
Q1 = data.quantile(0.35)
Q3 = data.quantile(0.65)
IQR = Q3 - Q1
#print (IQR)
data_out = data[~((data < (Q1-1.5*IQR)) | (data > (Q3 +1.5*IQR))).any(axis=1)]
#print (data_out)

#------------------Boxplot of the new data------------------
data_out.boxplot(['Memory(MB)', 'Time on local(s)', 'Time on Jetson(s)'])
plt.title('Boxplot of the new data')
plt.show()

#------------------Set data equals to new_data------------------
data = data_out.copy()

#--------------Correlation Matrix of the new dataset--------------
cor_matrix = data.corr()
sn.heatmap(cor_matrix, annot = True)
plt.title('Correlation Matrix of the New Dataset')
plt.show()

#--------------Construct the target column--------------
train_target = data['Time on Jetson(s)']

#--------------Exclude the target column from the dataset--------------
data.drop('Time on Jetson(s)', axis=1, inplace=True)

#--------------Split data-------------- 
X_train, X_test, y_train, y_test = train_test_split (data, train_target, 
                                        test_size=0.15, random_state=42)

""" print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape) """

# Let's try some Regression Models
#===============================================================================
#(1)- Linear Regression-------------- 
lr = LinearRegression()
lr.fit(X_train, y_train)

""" #--------------Predict training data--------------
y_pred_lr_train = lr.predict(X_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Linear Regression): ", 
        mean_squared_error(y_pred_lr_train, y_train))
 """
#--------------Predict test data--------------
y_pred_lr_test = lr.predict(X_test)

#--------------Evaluation Testing Data--------------
error_lr = mean_squared_error(y_pred_lr_test, y_test)
print ("MSE on Test Set (Linear Regression): ", error_lr)
#===============================================================================
#(2)- Random Forest Regression-------------- 
rf = RandomForestRegressor(n_estimators=20, max_depth=10)
rf.fit(X_train, y_train)

""" #--------------Predict training data--------------
y_pred_rf_train = rf.predict(X_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Random Forest Regression): ", 
        mean_squared_error(y_pred_rf_train, y_train))
 """
#--------------Predict test data--------------
y_pred_rf_test = rf.predict(X_test)

#--------------Evaluation Testing Data--------------
error_rf = mean_squared_error(y_pred_rf_test, y_test)
print ("MSE on Test Set (Random Forest Regression): ",error_rf)
#===============================================================================
#(3)- Decision Tree Regression-------------- 
dt = DecisionTreeRegressor(splitter='best', max_depth=5)
dt.fit(X_train, y_train)

""" #--------------Predict training data--------------
y_pred_dt_train = dt.predict(X_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Decision Tree Regression): ", 
        mean_squared_error(y_pred_dt_train, y_train))
 """
#--------------Predict test data--------------
y_pred_dt_test = dt.predict(X_test)

#--------------Evaluation Testing Data--------------
error_dt = mean_squared_error(y_pred_dt_test, y_test) 
print ("MSE on Test Set (Decision Tree Regression): ", error_dt)
#===============================================================================
#(4)- Ridge Regression-------------- 
rg = Ridge(alpha=1.0)
rg.fit(X_train, y_train)

""" #--------------Predict training data--------------
y_pred_rg_train = rg.predict(X_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Ridge Regression): ", 
        mean_squared_error(y_pred_rg_train, y_train))
 """
#--------------Predict test data--------------
y_pred_rg_test = rg.predict(X_test)

#--------------Evaluation Testing Data--------------
error_rg = mean_squared_error(y_pred_rg_test, y_test) 
print ("MSE on Test Set (Ridge Regression): ", error_rg) 
#===============================================================================
#(5)- Lasso Regression-------------- 
ls= Lasso(alpha=1.0)
ls.fit(X_train, y_train)

""" #--------------Predict training data--------------
y_pred_ls_train = ls.predict(X_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Lasso Regression): ", 
        mean_squared_error(y_pred_ls_train, y_train))
 """
#--------------Predict test data--------------
y_pred_ls_test = ls.predict(X_test)

#--------------Evaluation Testing Data--------------
error_ls = mean_squared_error(y_pred_ls_test, y_test)
print ("MSE on Test Set (Lasso Regression): ", error_ls)
#===============================================================================
#(6)- Polynomial Regression-------------- 
pl= PolynomialFeatures(degree=4)
X_poly_train= pl.fit_transform(X_train)
X_poly_test= pl.fit_transform(X_test)
pl = LinearRegression()
pl.fit(X_poly_train, y_train)

""" #--------------Predict training data--------------
y_pred_pl_train = pl.predict(X_poly_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Polynomial Regression): ", 
        mean_squared_error(y_pred_pl_train, y_train))
 """
#--------------Predict test data--------------
y_pred_pl_test = pl.predict(X_poly_test)

#--------------Evaluation Testing Data--------------
error_pl = mean_squared_error(y_pred_pl_test, y_test) 
print ("MSE on Test Set (Polynomial Regression): ", error_pl)
#===============================================================================
#(7)- Bayesian Ridge Regression-------------- 
br=BayesianRidge(compute_score=True) 
br.fit(X_train, y_train)

""" #--------------Predict training data--------------
y_pred_br_train = br.predict(X_train)

#--------------Evaluation Training Data--------------
print ("MSE on Training Set (Bayesian Ridge Regression): ", 
        mean_squared_error(y_pred_br_train, y_train))
 """
#--------------Predict test data--------------
y_pred_br_test = br.predict(X_test)

#--------------Evaluation Testing Data--------------
error_br = mean_squared_error(y_pred_br_test, y_test)
print ("MSE on Test Set (Bayesian Ridge Regression): ", error_br)
#===============================================================================
#--------------Bar Plot--------------
methods = ['Linear Regression', 'Random Forest Regression', 'Decision Tree Regression',
                'Ridge Regression', 'Lasso Regression', 'Polynomial Regression',
                'Bayesian Ridge Regression']
errors = [error_lr, error_rf, error_dt, error_rg, error_ls, error_pl, error_br]
y_pos = np.arange(len(methods))

plt.bar(y_pos, errors)
plt.xticks(y_pos, methods, rotation = 90, size = 8)
plt.subplots_adjust(bottom=0.35)
plt.ylim(0,150)
plt.title('Barplot of MS Errors')
plt.show()

#--------------Predicted and Actual Values--------------
plt.title('Predicted and Actual Values using Decision Tree Regression')
my_x_axis = []
for i in range(0,len(y_pred_dt_test)):
        my_x_axis.append(i)

plt.scatter(my_x_axis, y_test, color = 'red', label = 'Actual Values')
plt.scatter(my_x_axis, y_pred_dt_test, color = 'blue', label = 'Predicted Values')
plt.legend(loc = 'upper left')
plt.xlabel('Test Case')
plt.ylabel('Time on Jetson(s)')
plt.show()

#--------------Predicted and Actual Values Columns--------------
x = np.arange(len(my_x_axis))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x-width/2, y_test, width, label = "Actual Values")
rects2 = ax.bar(x+width/2, y_pred_dt_test, width, label = "Predicted Values")

ax.set_ylabel('Time on Jetson(s)')
ax.set_title('Predicted and Actual Values using Decision Tree Regression')
ax.set_xticks(x)
ax.set_xticklabels(my_x_axis)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()

#--------------Predicted and Actual Values Error--------------
plt.stem(my_x_axis, abs(y_pred_dt_test-y_test))
plt.title('Absolute Actual Error using Decision Tree Regression')
plt.ylabel('Time on Jetson(s)')
plt.xlabel('Test Case')
plt.show()
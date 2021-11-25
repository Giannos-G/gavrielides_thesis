import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#  Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=170)
regr_2 = DecisionTreeRegressor(max_depth=80)
regr_3 = DecisionTreeRegressor(max_depth=95)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

    #  Predict
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

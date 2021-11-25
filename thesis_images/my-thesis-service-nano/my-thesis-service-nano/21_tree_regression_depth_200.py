# tree_regression dependencies:
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#  Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=200)
regr_1.fit(X, y)

    #  Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)

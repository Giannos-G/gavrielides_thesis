# lasso_regression dependencies:
from sklearn.linear_model import Lasso
import numpy as np


#  Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
regr_1 = Lasso(alpha=1.0)
regr_2 = Lasso(alpha=25)
regr_3 = Lasso(alpha=115)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)
    
    #  Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

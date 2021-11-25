# lasso_regression dependencies:
from sklearn.linear_model import Lasso
import numpy as np

#  Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(2000000 * rng.rand(1000000, 1) - 1000000, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(200000, 2))

    #  Fit regression model
regr_1 = Lasso(alpha=90)
regr_2 = Lasso(alpha=180)
regr_1.fit(X, y)
regr_2.fit(X, y)

    #  Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

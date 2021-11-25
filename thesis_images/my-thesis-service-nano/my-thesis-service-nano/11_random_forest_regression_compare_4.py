# random_forest_regression dependencies:
from sklearn.ensemble import RandomForestRegressor
import numpy as np

    #  Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(20000 * rng.rand(10000, 1) - 10000, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(2000, 2))

    #  Fit regression model
regr_1 = RandomForestRegressor(n_estimators=20, max_depth=5)
regr_2 = RandomForestRegressor(n_estimators=20, max_depth=10)
regr_3 = RandomForestRegressor(n_estimators=20, max_depth=15)
regr_4 = RandomForestRegressor(n_estimators=20, max_depth=17)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)
regr_4.fit(X, y)

    #  Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)
y_4 = regr_4.predict(X_test)

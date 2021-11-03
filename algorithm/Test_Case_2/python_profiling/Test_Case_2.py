#  plot_det.py dependencies:
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_det_curve
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# plot_grid_search_digits.py dependencies:
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# plot_cv_digits.py dependencies:
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

# plot_feature_union.py dependencies:
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# plot_pca_vs_lda.py dependencies:
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# tree_regression dependencies:
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# tree_regression_mult dependencies:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# linear_regression dependencies:
from sklearn.linear_model import LinearRegression

# random_forest_regression dependencies:
from sklearn.ensemble import RandomForestRegressor

# ridge_regression dependencies:
from sklearn.linear_model import Ridge

# lasso_regression dependencies:
from sklearn.linear_model import Lasso

# bayesian_ridge_regression dependencies:
from sklearn.linear_model import BayesianRidge

# Time profiling dependencies:
import cProfile

#@profile
def plot_det():
    N_SAMPLES = 1000

    classifiers = {
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1),}

    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=2, n_redundant=0, n_informative=2,
        random_state=1, n_clusters_per_class=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=0)

    #  prepare plots
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        plot_roc_curve(clf, X_test, y_test, ax=ax_roc, name=name)
        plot_det_curve(clf, X_test, y_test, ax=ax_det, name=name)

    ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')
    ax_det.set_title('Detection Error Tradeoff (DET) curves')

    ax_roc.grid(linestyle='--')
    ax_det.grid(linestyle='--')

    plt.legend()
    # plt.show()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def plot_grid_search_digits():
    #  Loading the Digits dataset
    digits = datasets.load_digits()

    #  To apply an classifier on this data, we need to flatten the image, to
    #  turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    #  Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    #  Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        # # print("#  Tuning hyper-parameters for %s" % score)
        # # print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        # # print("Best parameters set found on development set:")
        # # print()
        # # print(clf.best_params_)
        # # print()
        # # print("Grid scores on development set:")
        # # print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            # # print("%0.3f (+/-%0.03f) for %r"
                # % (mean, std * 2, params))
        # # print()

        # # print("Detailed classification report:")
        # # print()
        # # print("The model is trained on the full development set.")
        # # print("The scores are computed on the full evaluation set.")
        # # print()
        y_true, y_pred = y_test, clf.predict(X_test)
        # # print(classification_report(y_true, y_pred))
        # # print()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def plot_cv_digits():
    X, y = datasets.load_digits(return_X_y=True)

    svc = svm.SVC(kernel='linear')
    C_s = np.logspace(-10, 0, 10)

    scores = list()
    scores_std = list()
    for C in C_s:
        svc.C = C
        this_scores = cross_val_score(svc, X, y, n_jobs=1)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))

    #  Do the plotting
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogx(C_s, scores)
    plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
    plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter C')
    plt.ylim(0, 1.1)
    # plt.show()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def plot_feature_union():
    iris = load_iris()

    X, y = iris.data, iris.target

    #  This dataset is way too high-dimensional. Better do PCA:
    pca = PCA(n_components=2)

    #  Maybe some original features were good, too?
    selection = SelectKBest(k=1)

    #  Build estimator from PCA and Univariate selection:

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    #  Use combined features to transform dataset:
    X_features = combined_features.fit(X, y).transform(X)
    # # print("Combined space has", X_features.shape[1], "features")

    svm = SVC(kernel="linear")

    #  Do grid search over k, n_components and C:

    pipeline = Pipeline([("features", combined_features), ("svm", svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3],
                    features__univ_select__k=[1, 2],
                    svm__C=[0.1, 1, 10])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    # grid_search.fit(X, y)
    # # print(grid_search.best_estimator_)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def plot_pca_vs_lda():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    #  Percentage of variance explained for each components
    # # print('explained variance ratio (first two components): %s'
        # % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')

    # plt.show()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def tree_regression_depth_200():
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def tree_regression_depth_150():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=150)
    regr_1.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def tree_regression_depth_170():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=170)
    regr_1.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def tree_regression_compare_2():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=200)
    regr_2 = DecisionTreeRegressor(max_depth=500)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def tree_regression_compare_3():
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def tree_regression_compare_4():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=200)
    regr_2 = DecisionTreeRegressor(max_depth=300)
    regr_3 = DecisionTreeRegressor(max_depth=40)
    regr_4 = DecisionTreeRegressor(max_depth=190)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)
    regr_4.fit(X, y)

    #  Predict
    X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)
    y_4 = regr_4.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def linear_regression():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(2000000 * rng.rand(1000000, 1) - 1000000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(200000, 2))

    #  Fit regression model
    regr_1 = LinearRegression()
    regr_1.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def random_forest_regression():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(20000 * rng.rand(10000, 1) - 10000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(2000, 2))

    #  Fit regression model
    regr_1 = RandomForestRegressor(n_estimators=20, max_depth=5)
    regr_1.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def random_forest_regression_compare_2():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(20000 * rng.rand(10000, 1) - 10000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(2000, 2))

    #  Fit regression model
    regr_1 = RandomForestRegressor(n_estimators=20, max_depth=5)
    regr_2 = RandomForestRegressor(n_estimators=20, max_depth=10)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def random_forest_regression_compare_3():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(20000 * rng.rand(10000, 1) - 10000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(2000, 2))

    #  Fit regression model
    regr_1 = RandomForestRegressor(n_estimators=20, max_depth=5)
    regr_2 = RandomForestRegressor(n_estimators=20, max_depth=10)
    regr_3 = RandomForestRegressor(n_estimators=20, max_depth=15)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def random_forest_regression_compare_4():
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def random_forest_regression_compare_5():
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
    regr_5 = RandomForestRegressor(n_estimators=20, max_depth=20)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)
    regr_4.fit(X, y)
    regr_5.fit(X, y)

    #  Predict
    X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)
    y_4 = regr_4.predict(X_test)
    y_5 = regr_5.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def ridge_regression():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(2000000 * rng.rand(1000000, 1) - 1000000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(200000, 2))

    #  Fit regression model
    regr_1 = Ridge(alpha=1.0)
    regr_1.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def ridge_regression_compare_2():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = Ridge(alpha=1.0)
    regr_2 = Ridge(alpha=2)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def ridge_regression_compare_3():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(2000000 * rng.rand(1000000, 1) - 1000000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(200000, 2))

    #  Fit regression model
    regr_1 = Ridge(alpha=1.0)
    regr_2 = Ridge(alpha=30)
    regr_3 = Ridge(alpha=10)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def ridge_regression_compare_4():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = Ridge(alpha=1.0)
    regr_2 = Ridge(alpha=30)
    regr_3 = Ridge(alpha=80)
    regr_4 = Ridge(alpha=86)
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def lasso_regression():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200000 * rng.rand(100000, 1) - 100000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20000, 2))

    #  Fit regression model
    regr_1 = Lasso(alpha=90)
    regr_1.fit(X, y)

    #  Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def lasso_regression_compare_2():
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def lasso_regression_compare_3():
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def lasso_regression_compare_4():
    #  Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(2000000 * rng.rand(1000000, 1) - 1000000, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(200000, 2))

    #  Fit regression model
    regr_1 = Lasso(alpha=100)
    regr_2 = Lasso(alpha=12)
    regr_3 = Lasso(alpha=17)
    regr_4 = Lasso(alpha=152)
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#@profile
def main():

    # Basic complete applications
    plot_det()
    # print ("plot_det() finished \n")
    plot_grid_search_digits()
    # print ("plot_grid_search_digits() finished \n")
    plot_cv_digits()
    # print ("plot_cv_digits() finished \n")
    plot_feature_union()
    # print ("plot_feature_union() finished \n")
    plot_pca_vs_lda()
    # print ("plot_pca_vs_lda() finished \n")

    # Tree regression models
    tree_regression_depth_200()
    # print ("tree_regression_depth_200() finished \n")
    tree_regression_depth_150()
    # print ("tree_regression_depth_3() finished \n")
    tree_regression_depth_170()
    # print ("tree_regression_depth_4() finished \n")
    tree_regression_compare_2()
    # print ("tree_regression_compare_2() finished \n")
    tree_regression_compare_3()
    # print ("tree_regression_compare_3() finished \n")
    tree_regression_compare_4()
    # print ("tree_regression_compare_4() finished \n")

    # Linear regression models
    linear_regression()
    # print ("linear_regression() finished \n")
    
    # Random Forest regression models
    random_forest_regression()
    # print ("random_forest_regression() finished \n")
    random_forest_regression_compare_2()
    # print ("random_forest_regression_compare_2() finished \n")
    random_forest_regression_compare_3()
    # print ("random_forest_regression_compare_3() finished \n")
    random_forest_regression_compare_4()
    # print ("random_forest_regression_compare_4() finished \n")
    random_forest_regression_compare_5()
    # print ("random_forest_regression_compare_5() finished \n")

    # Ridge regression models
    ridge_regression()
    # print ("ridge_regression() finished \n")
    ridge_regression_compare_2()
    # print ("ridge_regression_compare_2() finished \n")
    ridge_regression_compare_3()
    # print ("ridge_regression_compare_3() finished \n")
    ridge_regression_compare_4()
    # print ("ridge_regression_compare_4() finished \n")

    # Lasso regression models
    lasso_regression()
    # print ("lasso_regression() finished \n")
    lasso_regression_compare_2()
    # print ("lasso_regression_compare_2() finished \n")
    lasso_regression_compare_3()
    # print ("lasso_regression_compare_3() finished \n")
    lasso_regression_compare_4()
    # print ("lasso_regression_compare_4() finished \n")



if __name__ == '__main__':
    main()
    # cProfile.run('main()')
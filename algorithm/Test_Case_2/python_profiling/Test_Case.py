# plot_det.py dependencies:
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_det_curve
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#plot_grid_search_digits.py dependencies:
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#plot_cv_digits.py dependencies:
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

#plot_feature_union.py dependencies:
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#plot_pca_vs_lda.py dependencies:
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Python Memory Profiler
import cProfile

#Time Profiler
import cProfile

#@profile
def plot_det():
    N_SAMPLES = 1000

    classifiers = {
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1
        ),
    }

    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=2, n_redundant=0, n_informative=2,
        random_state=1, n_clusters_per_class=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=0)

    # prepare plots
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
    #plt.show()
##############################################################################

#@profile
def plot_grid_search_digits():
    # Loading the Digits dataset
    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
##############################################################################

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

    # Do the plotting
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
    #plt.show()
##############################################################################

#@profile
def plot_feature_union():
    iris = load_iris()

    X, y = iris.data, iris.target

    # This dataset is way too high-dimensional. Better do PCA:
    pca = PCA(n_components=2)

    # Maybe some original features were good, too?
    selection = SelectKBest(k=1)

    # Build estimator from PCA and Univariate selection:

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Use combined features to transform dataset:
    X_features = combined_features.fit(X, y).transform(X)
    print("Combined space has", X_features.shape[1], "features")

    svm = SVC(kernel="linear")

    # Do grid search over k, n_components and C:

    pipeline = Pipeline([("features", combined_features), ("svm", svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3],
                    features__univ_select__k=[1, 2],
                    svm__C=[0.1, 1, 10])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)
##############################################################################

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

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
        % str(pca.explained_variance_ratio_))

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

    #plt.show()

#@profile
def main():
  plot_det()
  plot_grid_search_digits()
  plot_cv_digits()
  plot_feature_union()
  plot_pca_vs_lda()


if __name__ == '__main__':
    main()
    #cProfile.run('main()')
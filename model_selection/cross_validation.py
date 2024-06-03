import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import numpy as np
from model_selection.kfold import KFold
from Estimator import Estimator

def cross_val_score(estimator: Estimator, X, y, cv=None):
    """
    Evaluate a score by cross-validation.

    Parameters:
        estimator : Estimator object implementing 'fit' and 'score'
            The object to use to fit the data.
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,)
            The target variable to try to predict in the case of supervised learning.
        cv : int, cross-validation generator, or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
                - None, to use the default 5-fold cross-validation
                - integer, to specify the number of folds in a KFold
                - An object to be used as a cross-validation generator
                - An iterable yielding train, test splits.
    
    Returns:
        scores : array of float, shape=(len(list(cv)),)
            Array of scores of the estimator for each run of the cross validation.
    """
    if cv is None:
        cv = KFold(n_splits=5, shuffle=False, random_state=None)

    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)
        scores.append(score)
    return np.array(scores)

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Predictor import Predictor
from supervised_learning.DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor

import numpy as np

class RandomForestClassifier(Predictor):
    """
    Random Forest Classifier class.

    Parameters:
    n_estimators (int): The number of trees in the forest (default=100).
    max_depth (int): The maximum depth of the tree (default=None).
    min_samples_split (int): The minimum number of samples required to split an internal node (default=2).
    min_samples_leaf (int): The minimum number of samples required to be at a leaf node (default=1).
    random_state (int): Controls the randomness of the bootstrapping of the samples used when building trees (default=None).
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
            tree.fit(X, y)
            self.estimators_.append(tree)

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        X (array-like): Input data.

        Returns:
        array-like: Predicted class labels.
        """
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        # Calculate the most common prediction for each sample
        return np.array([np.argmax(np.bincount(tree_predictions)) for tree_predictions in predictions.T])

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        X (array-like): Test data.
        y (array-like): True labels for X.

        Returns:
        float: Mean accuracy of the classifier.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class RandomForestRegressor(Predictor):
    """
    Random Forest Regressor class.

    Parameters:
    n_estimators (int): The number of trees in the forest (default=100).
    max_depth (int): The maximum depth of the tree (default=None).
    min_samples_split (int): The minimum number of samples required to split an internal node (default=2).
    min_samples_leaf (int): The minimum number of samples required to be at a leaf node (default=1).
    random_state (int): Controls the randomness of the bootstrapping of the samples used when building trees (default=None).
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
            tree.fit(X, y)
            self.estimators_.append(tree)

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters:
        X (array-like): Input data.

        Returns:
        array-like: Predicted target values.
        """
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        X (array-like): Test data.
        y (array-like): True values for X.

        Returns:
        float: R^2 score.
        """
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

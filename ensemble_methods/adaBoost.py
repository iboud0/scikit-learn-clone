import numpy as np
import os
import sys
from copy import deepcopy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Estimator import Estimator
from metrics_model_evaluation.accuracy import accuracy_score

# Utility functions
def check_is_fitted(estimator, attributes):
    """
    Check if an estimator is fitted by verifying the presence of specified attributes.
    
    Parameters:
    estimator : object
        The estimator instance to check.
    attributes : str or list of str
        The attribute or attributes to check for.
        
    Raises:
    ValueError
        If any of the specified attributes are not found in the estimator.
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(f"This {type(estimator).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
def clone(estimator):
    """
    Create a deep copy of an estimator.
    
    Parameters:
    estimator : object
        The estimator instance to clone.
        
    Returns:
    object
        A deep copy of the estimator.
    """
    return deepcopy(estimator)

# ClassifierMixin for scoring
class ClassifierMixin:
    """
    Mixin class for all classifiers in scikit-learn.
    """

    def score(self, X, y):
        """
        Return the accuracy score on the given test data and labels.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.
        
        Returns:
        float
            Accuracy score.
        """
        return accuracy_score(y, self.predict(X))

# RegressorMixin for scoring
class RegressorMixin:
    """
    Mixin class for all regressors in scikit-learn.
    """
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True values for X.
        
        Returns:
        float
            R^2 score.
        """
        return self._score(X, y)
    
    def _score(self, X, y):
        """
        Calculate the R^2 score.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True values for X.
        
        Returns:
        float
            R^2 score.
        """
        return 1 - np.sum((y - self.predict(X))**2) / np.sum((y - np.mean(y))**2)

# AdaBoostClassifier
class AdaBoostClassifier(Estimator, ClassifierMixin):
    """
    An AdaBoost classifier.
    
    Parameters:
    base_estimator : object
        The base estimator from which the boosted ensemble is built.
    n_estimators : int, optional (default=10)
        The maximum number of estimators at which boosting is terminated.
    random_state : int or RandomState instance, optional (default=None)
        Controls the random seed given at each base_estimator at each boosting iteration.
    """
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_weights = np.zeros(n_estimators)
        self.estimators_ = []

    def fit(self, X, y):
        """
        Build a boosted classifier from the training set (X, y).
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels).
        
        Returns:
        self : object
            Returns self.
        """
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred = estimator.predict(X)

            incorrect = y_pred != y
            error = np.sum(sample_weight * incorrect)

            if error > 0.5:
                break
            if error == 0:
                alpha = 0
            else:
                alpha = np.log((1 - error) / error) / 2
            
            sample_weight *= np.exp(-alpha * y * y_pred)
            sample_weight /= np.sum(sample_weight)

            self.estimators_.append(estimator)
            self.estimator_weights[i] = alpha

        return self

    def predict(self, X, y=None):
        """
        Predict classes for X.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,), optional (default=None)
            The target values (class labels). Only used to determine the number of classes.
        
        Returns:
        y_pred : array, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, ['estimators_', 'estimator_weights'])
        n_classes = len(np.unique(y))
        n_estimators = len(self.estimators_)
        class_counts = np.zeros((X.shape[0], n_classes))

        for i, estimator in enumerate(self.estimators_):
            y_pred = estimator.predict(X)
            for j in range(n_classes):
                class_counts[:, j] += (y_pred == j) * self.estimator_weights[i]

        return np.argmax(class_counts, axis=1)

# AdaBoostRegressor
class AdaBoostRegressor(Estimator, RegressorMixin):
    """
    An AdaBoost regressor.
    
    Parameters:
    base_estimator : object
        The base estimator from which the boosted ensemble is built.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
    random_state : int or RandomState instance, optional (default=None)
        Controls the random seed given at each base_estimator at each boosting iteration.
    """
    def __init__(self, base_estimator, n_estimators=50, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_weights = np.zeros(n_estimators)
        self.estimators_ = []

    def fit(self, X, y):
        """
        Build a boosted regressor from the training set (X, y).
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (real numbers).
        
        Returns:
        self : object
            Returns self.
        """
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred = estimator.predict(X)

            residuals = y - y_pred
            error = np.sum(sample_weight * np.abs(residuals)) / np.sum(sample_weight)

            if error >= 0.5:
                break
            if error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            sample_weight *= np.exp(alpha * residuals)
            sample_weight /= np.sum(sample_weight)

            self.estimators_.append(estimator)
            self.estimator_weights[i] = alpha

        return self

    def predict(self, X):
        """
        Predict regression target for X.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input samples.
        
        Returns:
        y_pred : array, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ['estimators_', 'estimator_weights'])

        y_pred = np.zeros(X.shape[0])
        for i, estimator in enumerate(self.estimators_):
            y_pred += self.estimator_weights[i] * estimator.predict(X)

        return y_pred

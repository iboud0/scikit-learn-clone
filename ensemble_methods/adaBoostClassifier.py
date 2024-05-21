import numpy as np
import os
import sys
from copy import deepcopy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Estimator import Estimator
from metrics.accuracy import accuracy_score

class ClassifierMixin:
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def check_is_fitted(estimator, attributes):
    if isinstance(attributes, str):
        attributes = [attributes]
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(f"This {type(estimator).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
def clone(estimator):
    return deepcopy(estimator)

class AdaBoostClassifier(Estimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_weights = np.zeros(n_estimators)
        self.estimators_ = []

    def fit(self, X, y):
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
        check_is_fitted(self, ['estimators_', 'estimator_weights'])
        n_classes = len(np.unique(y))
        n_estimators = len(self.estimators_)
        class_counts = np.zeros((X.shape[0], n_classes))

        for i, estimator in enumerate(self.estimators_):
            y_pred = estimator.predict(X)
            for j in range(n_classes):
                class_counts[:, j] += (y_pred == j) * self.estimator_weights[i]

        return np.argmax(class_counts, axis=1)


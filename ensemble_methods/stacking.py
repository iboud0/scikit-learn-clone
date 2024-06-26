import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from Estimator import Estimator
from Predictor import Predictor
from model_selection import KFold

class StackingRegressor(Predictor):
    """
    A Stacking Regressor for combining multiple regression models.

    Parameters:
    base_models (list): List of base models to be used for stacking.
    meta_model (object): Meta-model used to aggregate the predictions of the base models.
    n_folds (int): Number of folds for cross-validation (default=5).
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        """
        Fit the Stacking Regressor model.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        self.X_meta = np.zeros((X.shape[0], len(self.base_models)))

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                self.X_meta[holdout_idx, i] = instance.predict(X[holdout_idx])

        self.meta_model_.fit(self.X_meta, y)

    def predict(self, X):
        """
        Predict target values using the fitted model.

        Parameters:
        X (array-like): Input data.

        Returns:
        array-like: Predicted target values.
        """
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

# Helper function to clone a model
def clone(estimator: Estimator):
    """
    Clone an estimator by creating a new instance with the same parameters.

    Parameters:
    estimator (Estimator): The estimator to clone.

    Returns:
    Estimator: A new instance of the estimator with the same parameters.
    """
    return estimator.__class__(**estimator.get_params())

class StackingClassifier(Predictor):
    """
    A Stacking Classifier for combining multiple classification models.

    Parameters:
    base_models (list): List of base models to be used for stacking.
    meta_model (object): Meta-model used to aggregate the predictions of the base models.
    n_folds (int): Number of folds for cross-validation (default=5).
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        """
        Fit the Stacking Classifier model.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        self.X_meta = np.zeros((X.shape[0], len(self.base_models)))

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                self.X_meta[holdout_idx, i] = instance.predict(X[holdout_idx])

        self.meta_model_.fit(self.X_meta, y)

    def predict(self, X):
        """
        Predict class labels using the fitted model.

        Parameters:
        X (array-like): Input data.

        Returns:
        array-like: Predicted class labels.
        """
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

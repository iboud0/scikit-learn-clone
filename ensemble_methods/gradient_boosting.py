import numpy as np
from Predictor import Predictor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

class GradientBoostingRegressor(Predictor):
    """
    Gradient Boosting Regressor class.

    Parameters:
    n_estimators (int): The number of boosting stages to perform (default=100).
    learning_rate (float): The learning rate shrinks the contribution of each tree by learning_rate (default=0.1).
    max_depth (int): The maximum depth of the individual regression estimators (default=3).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.loss = mean_squared_error

    def fit(self, X, y):
        """
        Fit the gradient boosting model.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        # Initialize the predictions with the mean of y
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            # Compute the residuals (negative gradients)
            residuals = y - predictions

            # Fit a new tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            predictions += self.learning_rate * tree.predict(X)

    def predict(self, X):
        """
        Predict using the gradient boosting model.

        Parameters:
        X (array-like): Input data.
        
        Returns:
        array-like: Predicted values.
        """
        # Start with the initial prediction
        predictions = np.full((X.shape[0],), self.initial_prediction)

        # Add the contributions from all the trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

class GradientBoostingClassifier(Predictor):
    """
    Gradient Boosting Classifier class.

    Parameters:
    n_estimators (int): The number of boosting stages to perform (default=100).
    learning_rate (float): The learning rate shrinks the contribution of each tree by learning_rate (default=0.1).
    max_depth (int): The maximum depth of the individual classification estimators (default=3).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        # Initialize with the log-odds
        positive_class_prob = np.sum(y) / len(y)
        self.initial_pred = np.log(positive_class_prob / (1 - positive_class_prob))
        predictions = np.full(y.shape, self.initial_pred)

        for _ in range(self.n_estimators):
            # Compute the gradients (negative gradients for binary log loss)
            gradients = y - self._sigmoid(predictions)

            # Convert gradients to binary classes
            classes = np.where(gradients >= 0.5, 1, 0)

            # Fit a new tree to the classes
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, classes)
            self.trees.append(tree)

            # Update predictions
            predictions += self.learning_rate * tree.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        X (array-like): Input data.
        
        Returns:
        array-like: Predicted probabilities.
        """
        # Start with the initial prediction
        predictions = np.full((X.shape[0],), self.initial_pred)

        # Add the contributions from all the trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        probas = self._sigmoid(predictions)
        return np.vstack([1 - probas, probas]).T

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        X (array-like): Input data.
        
        Returns:
        array-like: Predicted class labels.
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
    
    def _sigmoid(self, x):
        """
        Sigmoid function.

        Parameters:
        x (array-like): Input data.
        
        Returns:
        array-like: Sigmoid of the input data.
        """
        return 1 / (1 + np.exp(-x))

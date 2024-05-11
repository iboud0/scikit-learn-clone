
from Predictor import Predictor
import numpy as np

class LogisticRegression(Predictor):
    """
    Logistic regression predictor.

    Attributes:
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    max_iters : int, default=1000
        Maximum number of iterations for gradient descent.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    weights : array-like, shape (n_features,)
        Coefficients of the features.
    bias : float
        Intercept term.

    Methods:
    --------
    __init__(learning_rate=0.01, max_iters=1000, tol=1e-4):
        Initialize logistic regression predictor.
    fit(X, y):
        Fit the model according to the given training data.
    predict(X):
        Predict the target for the provided data.
    score(X, y):
        Return the coefficient of determination R^2 of the prediction.
    """

    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-4):
        """
        Initialize logistic regression predictor.

        Parameters:
        ----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent.
        max_iters : int, default=1000
            Maximum number of iterations for gradient descent.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        """
        Compute the sigmoid function.

        Parameters:
        -----------
        x : array-like
            Input values.

        Returns:
        --------
        sigmoid : array-like
            Output values after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Training samples.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        -------
        self : LogisticRegression
            Fitted logistic regression predictor.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        prev_cost = float('inf')

        for _ in range(self.max_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            cost = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))

            if abs(prev_cost - cost) < self.tol:
                break

            prev_cost = cost
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self

    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Data samples.

        Returns:
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted targets.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = np.round(y_predicted)
        return y_predicted_cls

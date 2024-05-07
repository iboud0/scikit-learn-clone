from Estimator import Estimator

class Predictor(Estimator):
    """
    Base class for predictors in the API.

    Predictors in this API implement methods for making predictions and
    evaluating prediction accuracy.

    Attributes:
    ----------
    None

    Methods:
    --------
    predict(X):
        Predict the target for the provided data.

    fit(X, y):
        Fit the model according to the given training data.

    fit_predict(X, y=None):
        Fit the model and predict the target for the provided data.

    score(X, y):
        Return the coefficient of determination R^2 of the prediction.
    """
    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Data to predict.

        Returns:
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted targets.
        """
        pass

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        -------
        self : Predictor
            Fitted predictor.
        """
        pass

    def fit_predict(self, X, y=None):
        """
        Fit the model and predict the target for the provided data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Data.
        y : array-like, shape (n_samples,), optional
            Target values.

        Returns:
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted targets.
        """
        pass

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.

        Returns:
        -------
        score : float
            R^2 of self.predict(X) with respect to y.
        """
        pass
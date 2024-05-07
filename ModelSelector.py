from MetaEstimator import MetaEstimator

class ModelSelector(MetaEstimator):
    """
    Class for selecting the best model from a set of hyperparameters.

    Model selectors in this API train the estimator multiple times with
    different values for the hyperparameters when the fit method is called.

    Attributes:
    ----------
    None

    Methods:
    --------
    best_score():
        Return the best score achieved during model selection.

    best_params():
        Return the parameters that resulted in the best score.

    best_estimator():
        Return the estimator that achieved the best score.

    fit(X, y):
        Fit the model selector according to the given training data.

    fit_predict(X, y=None):
        Fit the model selector and predict the target for the provided data.

    score(X, y):
        Return the coefficient of determination R^2 of the prediction.
    """
    def best_score(self):
        """
        Return the best score achieved during model selection.

        Returns:
        -------
        best_score : float
            Best score achieved during model selection.
        """
        pass

    def best_params(self):
        """
        Return the parameters that resulted in the best score.

        Returns:
        -------
        best_params : dict
            Parameters that resulted in the best score.
        """
        pass

    def best_estimator(self):
        """
        Return the estimator that achieved the best score.

        Returns:
        -------
        best_estimator : Estimator
            Estimator that achieved the best score.
        """
        pass

    def fit(self, X, y):
        """
        Fit the model selector according to the given training data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        -------
        self : ModelSelector
            Fitted model selector.
        """
        pass

    def fit_predict(self, X, y=None):
        """
        Fit the model selector and predict the target for the provided data.

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
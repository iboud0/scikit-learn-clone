from MetaEstimator import MetaEstimator

class Pipeline(MetaEstimator):
    """
    Class for constructing pipelines of transformers and a final predictor.

    Pipelines in this API combine multiple transformers and a final predictor
    into a single estimator.

    Attributes:
    ----------
    None

    Methods:
    --------
    fit(X, y=None):
        Fit the pipeline according to the given training data.

    fit_predict(X, y=None):
        Fit the pipeline and predict the target for the provided data.

    score(X, y):
        Return the coefficient of determination R^2 of the prediction.
    """
    def fit(self, X, y=None):
        """
        Fit the pipeline according to the given training data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Target values.

        Returns:
        -------
        self : Pipeline
            Fitted pipeline.
        """
        pass

    def fit_predict(self, X, y=None):
        """
        Fit the pipeline and predict the target for the provided data.

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
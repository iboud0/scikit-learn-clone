class MetaEstimator:
    """
    Base class for meta-estimators in the API.

    Meta-estimators in this API combine one or more base estimators into a single
    estimator.

    Attributes:
    ----------
    estimators : list
        List of base estimators to be combined.

    Methods:
    --------
    fit(X, y=None):
        Fit the meta-estimator according to the given training data.
    """
    def __init__(self, estimators):
        """
        Initialize the meta-estimator with a list of base estimators.

        Parameters:
        ----------
        estimators : list
            List of base estimators to be combined.
        """
        self.estimators = estimators

    def fit(self, X, y=None):
        """
        Fit the meta-estimator according to the given training data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Target values.

        Returns:
        -------
        self : MetaEstimator
            Fitted meta-estimator.
        """
        pass
    
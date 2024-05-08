class Estimator:
    """
    Base class for all estimators in the API.

    Estimators in this API implement a common interface for fitting models
    to data and making predictions.

    Attributes:
    ----------
    None

    Methods:
    --------
    get_params(deep=True):
        Get parameters for this estimator.

    set_params(**params):
        Set the parameters of this estimator.

    fit(X, y=None):
        Fit the model according to the given training data.
    """
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.__dict__

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters:
        ----------
        **params : dict
            Estimator parameters.

        Returns:
        -------
        self : Estimator
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Target values.

        Returns:
        -------
        self : Estimator
            Fitted estimator.
        """
        pass
    
from Estimator import Estimator

class Transformer(Estimator):
    """
    Base class for transformers in the API.

    Transformers in this API implement methods for preprocessing data.

    Attributes:
    ----------
    None

    Methods:
    --------
    transform(X):
        Transform the input data.

    fit_transform(X, y=None):
        Fit to data, then transform it.
    """
    def transform(self, X):
        """
        Transform the input data.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Data to transform.

        Returns:
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        pass

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Target values.

        Returns:
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        pass

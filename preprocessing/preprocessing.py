import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def scale_features(self, X):
        """
        Scale features in X using Min-Max scaling.
        """
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_scaled = (X - X_min) / (X_max - X_min)
        return X_scaled

    def normalize_features(self, X):
        """
        Normalize features in X to have unit norm.
        """
        norms = np.linalg.norm(X, axis=1)
        X_normalized = X / norms[:, None]
        return X_normalized

    def impute_missing_values_mean(self, X):
        """
        Impute missing values in X using mean.
        """
        missing_mask = np.isnan(X)
        X_imputed = np.where(missing_mask, np.nanmean(X, axis=0), X)
        return X_imputed
    
    def impute_missing_values_median(self, X):
        """
        Impute missing values in X using median.
        """
        missing_mask = np.isnan(X)
        X_imputed = np.where(missing_mask, np.nanmedian(X, axis=0), X)
        return X_imputed
    


    def encode_categorical_variables_one_hot(self, X):
        """
        Encode categorical variables in X using one-hot encoding.
        """
        unique_values = np.unique(X)
        encoded = np.eye(len(unique_values))[np.searchsorted(unique_values, X)]
        return encoded
    
    def encode_categorical_variables_label(self, X):
        """
        Encode categorical variables in X using label encoding.
        """
        unique_values = np.unique(X)
        encoded = np.searchsorted(unique_values, X)
        return encoded

    def select_features(self, X, y, k=5):
        """
        Select a subset of features in X using SelectKBest and f_classif.
        """
        scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            scores[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        top_indices = np.argsort(scores)[::-1][:k]
        X_selected = X[:, top_indices]
        return X_selected
    


#test
if __name__ == '__main__':
    # Testing Preprocessor
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    preprocessor = Preprocessor()
    print(preprocessor.scale_features(X)) # Expected output: [[0. 0. 0.] [0.5 0.5 0.5] [1. 1. 1.]]
    print(preprocessor.normalize_features(X)) # Expected output: [[0.26726124 0.53452248 0.80178373] [0.45584231 0.56980288 0.68376346] [0.50257071 0.57436653 0.64616234]]
    print(preprocessor.impute_missing_values_mean(X)) # Expected output: [[1. 2. 3.] [4. 5. 6.] [7. 8. 9.]]
    print(preprocessor.impute_missing_values_median(X)) # Expected output: [[1. 2. 3.] [4. 5. 6.] [7. 8. 9.]]
    print(preprocessor.encode_categorical_variables_one_hot(np.array(['a', 'b', 'c', 'a']))) # Expected output: [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.] [1. 0. 0.]]
    print(preprocessor.encode_categorical_variables_label(np.array(['a', 'b', 'c', 'a']))) # Expected output: [0 1 2 0]
    print(preprocessor.select_features(X, y, k=2)) # Expected output: [[2 3] [5 6] [8 9]]

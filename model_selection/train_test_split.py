import numpy as np

class train_test_split:
    def __init__(self, random_state=None):
        """
        Initialize the DataSplitter.

        Parameters:
        - random_state: Seed for the random number generator (default is None)
        """
        self.random_state = random_state

    def split(self, X, y, test_size=0.2):
        """
        Split the data into training and testing sets.

        Parameters:
        - X: Features (numpy array or list of lists)
        - y: Target variable (numpy array or list)
        - test_size: Proportion of the data to include in the test split (default is 0.2)

        Returns:
        - X_train: Features for training
        - X_test: Features for testing
        - y_train: Target variable for training
        - y_test: Target variable for testing
        """
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples = len(X)
        test_indices = np.random.choice(n_samples, size=int(test_size * n_samples), replace=False)
        train_indices = np.array([i for i in range(n_samples) if i not in test_indices])

        X_train = np.array([X[i] for i in train_indices])
        X_test = np.array([X[i] for i in test_indices])
        y_train = np.array([y[i] for i in train_indices])
        y_test = np.array([y[i] for i in test_indices])

        return X_train, X_test, y_train, y_test 
#main
if __name__ == '__main__':
    # Testing DataSplitter
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    splitter = train_test_split(random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y, test_size=0.2)
    print(X_train) # Expected output: [[5, 6], [1, 2], [9, 10], [3, 4]]
    print(X_test) # Expected output: [[7, 8]]
    print(y_train) # Expected output: [0, 0, 0, 1]
    print(y_test) # Expected output: [1]
          

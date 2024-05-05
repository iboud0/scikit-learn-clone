import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
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

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = np.round(y_predicted)
        return y_predicted_cls


#main

if __name__ == '__main__':
    # Testing Logistic Regression
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression()
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [1.]
          )
    # Testing Logistic Regression with different learning rate
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression(learning_rate=0.1)
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [1.]
          )
    # Testing Logistic Regression with different max iterations
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression(max_iters=10000)
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [1.]
          )
    # Testing Logistic Regression with different tolerance
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression(tol=1e-6)
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [1.]
          )
    # Testing Logistic Regression with multiple features
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression()
    lr.fit(X, y)
    print(lr.predict(np.array([[6, 7]])) # Expected output: [1.]
          )
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    lr.fit(X, y)
    print(lr.predict(np.array([[6, 7], [7, 8], [8, 9]])) # Expected output: [1. 1. 1.]
        )
          

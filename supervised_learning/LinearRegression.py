import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, reg_strength=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.reg_strength = reg_strength
    
    def fit(self, X, y):
        # samples and features
        n_samples, n_features = X.shape
        
        # weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (model - y))
            db = (1 / n_samples) * np.sum(model - y)
            
            if self.regularization == 'l1':
                dw += self.reg_strength * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += self.reg_strength * self.weights
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    


#main
if __name__ == '__main__':
    # Testing Linear Regression
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    lr = LinearRegression()
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [12.]
          )
    # Testing Linear Regression with L1 regularization
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    lr = LinearRegression(regularization='l1', reg_strength=0.01)
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [12.]
          )
    # Testing Linear Regression with L2 regularization
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    lr = LinearRegression(regularization='l2', reg_strength=0.01)
    lr.fit(X, y)
    print(lr.predict(np.array([[6]])) # Expected output: [12.]
          )
    

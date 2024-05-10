import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])
    def _grow_tree(self, X, y, depth=0, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return Node(value=np.argmax([np.sum(y == i) for i in range(self.n_classes_)]))

        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(n_samples_per_class)
        node = Node(value=predicted_class)

        if max_depth is None or depth < max_depth:
            feature, threshold = self._best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = feature
                node.threshold = threshold
                node.left = self._grow_tree(X_left, y_left, depth + 1, max_depth)
                node.right = self._grow_tree(X_right, y_right, depth + 1, max_depth)

        return node



    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_feature, best_threshold = None, None

        for feature in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_feature, best_threshold

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class DecisionTreeRegressor(DecisionTree):
    def fit(self, X, y):
        self.y_mean_ = np.mean(y)
        super().fit(X, y)

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class DecisionTreeClassifier(DecisionTree):
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

# Example usage
X = np.array([[0, 0], [1, 1], [2, 2]])
y_reg = np.array([0, 1, 2])
y_clf = np.array([0, 0, 1])

dt_reg = DecisionTreeRegressor()
dt_clf = DecisionTreeClassifier()
dt_reg.fit(X, y_reg)
dt_clf.fit(X, y_clf)

print("Regression Prediction:", dt_reg.predict([[1, 0]]))
print("Classification Prediction:", dt_clf.predict([[1, 0]]))

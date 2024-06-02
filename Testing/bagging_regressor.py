
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from metrics_model_evaluation.accuracy import accuracy_score

# Test script
import unittest
from ensemble_methods.bagging import BaggingRegressor
from supervised_learning.DecisionTree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor as SklearnBaggingRegressor
from sklearn.datasets import load_iris
from model_selection.train_test_split import TrainTestSplitter

class TestBaggingRegressor(unittest.TestCase):
    def test_bagging_regressor(self):
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split dataset
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # Create and fit BaggingRegressor
        base_estimator = DecisionTreeRegressor(max_depth=3)
        bagging_regressor = BaggingRegressor(base_estimator=base_estimator, n_estimators=5, random_state=42)
        bagging_regressor.fit(X_train, y_train)

        # Evaluate BaggingRegressor
        score = bagging_regressor.score(X_test, y_test)

        # Compare with sklearn's BaggingRegressor
        sk_bagging_regressor = SklearnBaggingRegressor( n_estimators=5, random_state=42)
        sk_bagging_regressor.fit(X_train, y_train)
        sk_score = sk_bagging_regressor.score(X_test, y_test)

        # Assert that the scores are equal
        self.assertAlmostEqual(score, sk_score, places=5)

if __name__ == '__main__':
    unittest.main()

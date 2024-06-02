import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from ensemble_methods.adaBoost import AdaBoostRegressor

class TestAdaBoostRegressor(unittest.TestCase):
    def test_adaBoost_regressor(self):
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and fit AdaBoostRegressor
        base_estimator = DecisionTreeRegressor(max_depth=3)
        adaboost_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=5, random_state=42)
        adaboost_regressor.fit(X_train, y_train)

        # Evaluate AdaBoostRegressor
        score = adaboost_regressor.score(X_test, y_test)

        # Compare with scikit-learn's AdaBoostRegressor
        from sklearn.ensemble import AdaBoostRegressor as SklearnAdaBoostRegressor
        sk_adaboost_regressor = SklearnAdaBoostRegressor(n_estimators=5, random_state=42)
        sk_adaboost_regressor.fit(X_train, y_train)
        sk_score = sk_adaboost_regressor.score(X_test, y_test)

        # Assert that the scores are equal
        self.assertAlmostEqual(score, sk_score, places=5)

if __name__ == '__main__':
    unittest.main()
 



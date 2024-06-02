import numpy as np
import unittest
import os
import sys
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import your custom GradientBoosting implementations
from ensemble_methods.gradient_boosting import GradientBoostingRegressor as CustomGBR
from ensemble_methods.gradient_boosting import GradientBoostingClassifier as CustomGBC

# Import scikit-learn's implementations for comparison
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC

class TestGradientBoosting(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()

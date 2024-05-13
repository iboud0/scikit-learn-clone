# Testing/test_onehotencoder.py
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import unittest
import numpy as np
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder
from preprocessing.onehotencoder import OneHotEncoder


class TestOneHotEncoder(unittest.TestCase):
    def test_fit_transform_custom(self):
        X_train = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1]])
        encoder_custom = OneHotEncoder(feature_name_combiner=lambda x, y: f'x{x}_{y}')
        encoder_custom.fit(X_train)
        X_encoded_custom = encoder_custom.transform(X_train)

        self.assertEqual(X_encoded_custom.shape, (4, 6))  # Expected shape after custom one-hot encoding

    # def test_fit_transform_sklearn(self):
    #     X_train = np.array([[0], [1], [1], [0]])  # Modify input format to match sklearn's OneHotEncoder
    #     encoder_sklearn = SKOneHotEncoder(drop='first', sparse=False)
    #     X_encoded_sklearn = encoder_sklearn.fit_transform(X_train)

    #     self.assertEqual(X_encoded_sklearn.shape, (4, 1))  # Expected shape after sklearn one-hot encoding

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
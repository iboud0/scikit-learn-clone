# # Testing/test_select_features.py
# import sys
# sys.path.append('C:\\Users\\Pc\\Downloads\\scikit-learn-clone')
# import unittest
# import numpy as np
# from preprocessing.select_features import select_features

# from sklearn.feature_selection import SelectKBest, f_regression
# class TestSelectFeatures(unittest.TestCase):

#     def test_correlation(self):
#         # Generate some dummy data
#         np.random.seed(0)
#         X = np.random.rand(100, 10)
#         y = np.random.rand(100)

#         # Select features using your function
#         X_selected_custom = select_features(X, y, k=5, method='correlation')

#         # Select features using scikit-learn's implementation
#         selector = SelectKBest(score_func=f_regression, k=5)
#         X_selected_sklearn = selector.fit_transform(X, y)

#         # Print selected features and their scores
#         print("Custom function:")
#         print(X_selected_custom)
#         print("Scikit-learn:")
#         print(X_selected_sklearn)

#         # Compare the results
#         self.assertTrue(np.array_equal(X_selected_custom, X_selected_sklearn))

# if __name__ == '__main__':
#     unittest.main()
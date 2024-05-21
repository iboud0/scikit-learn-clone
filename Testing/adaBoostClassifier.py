import unittest
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from ensemble_methods.adaBoostClassifier import AdaBoostClassifier
from supervised_learning.DecisionTree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier
from sklearn.datasets import load_iris
from model_selection.train_test_split import TrainTestSplitter
from metrics.accuracy import accuracy_score

class TestAdaBoostClassifier(unittest.TestCase):
    def setUp(self):
        self.data = load_iris()
        splitter = TrainTestSplitter(random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = splitter.split(self.data.data, self.data.target)
        self.base_estimator = DecisionTreeClassifier()

    def test_adaboost_classifier(self):
        my_adaboost = AdaBoostClassifier(base_estimator=self.base_estimator, n_estimators=10, random_state=42)
        my_adaboost.fit(self.X_train, self.y_train)
        my_preds = my_adaboost.predict(self.X_test)
        my_accuracy = accuracy_score(self.y_test, my_preds)

        sklearn_adaboost = SklearnAdaBoostClassifier(n_estimators=10)
        sklearn_adaboost.fit(self.X_train, self.y_train)
        sklearn_preds = sklearn_adaboost.predict(self.X_test)
        sklearn_accuracy = accuracy_score(self.y_test, sklearn_preds)

        print(f"My AdaBoostClassifier accuracy: {my_accuracy}")
        print(f"Scikit-learn AdaBoostClassifier accuracy: {sklearn_accuracy}")

        self.assertAlmostEqual(my_accuracy, sklearn_accuracy, places=2)

if __name__ == '__main__':
    unittest.main()

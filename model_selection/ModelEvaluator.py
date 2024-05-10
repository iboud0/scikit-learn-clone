import numpy as np
import sys
sys.path.append('C:\\Users\\Pc\\Downloads\\scikit-learn-clone')

from supervised_learning.LinearRegression import LinearRegression
from supervised_learning.LogisticRegression import LogisticRegression
from supervised_learning.knn import KNNClassifier
from supervised_learning.NaiveBayes import GaussianNB
from supervised_learning.decisiontrees import DecisionTree


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the model.
        """
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(self, y_true, y_pred):
        """
        Calculate the precision of the model.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall(self, y_true, y_pred):
        """
        Calculate the recall of the model.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1_score(self, y_true, y_pred):
        """
        Calculate the F1 score of the model.
        """
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def roc_auc_score(self, y_true, y_pred):
        """
        Calculate the ROC-AUC score of the model.
        """
        # This implementation assumes binary classification
        n = len(y_true)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Calculate the area under the ROC curve using the trapezoidal rule
        tprs = [0] + [tpr] + [1]
        fprs = [0] + [fpr] + [1]
        area = np.sum([(fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2 for i in range(len(tprs) - 1)])

        return area

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate the mean squared error of the model.
        """
        return np.mean((y_true - y_pred) ** 2)

    def evaluate(self, X, y):
        """
        Evaluate the model using various metrics.
        """
        y_pred = self.model.predict(X)
        y_pred_cls = np.round(y_pred)

        accuracy = self.accuracy(y, y_pred_cls)
        precision = self.precision(y, y_pred_cls)
        recall = self.recall(y, y_pred_cls)
        f1 = self.f1_score(y, y_pred_cls)
        roc_auc = self.roc_auc_score(y, y_pred)
        mse = self.mean_squared_error(y, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'mse': mse
        }


#test
if __name__ == '__main__':
    # Testing ModelEvaluator
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    model = LinearRegression()
    model.fit(X, y)
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X, y)
    print(metrics) 
    from sklearn.linear_model import LinearRegression 
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_y_pred = sklearn_model.predict(X)

    # Calculating metrics for scikit-learn Linear Regression
    sklearn_metrics = {
        'mse': np.mean((y - sklearn_y_pred) ** 2)
    }
    print("\nScikit-learn Linear Regression Metrics:")
    print(sklearn_metrics)                                                                   
          
    # Expected output: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0, 'mse': 0.0}
    # Testing ModelEvaluator with Logistic Regression
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    model1 = LogisticRegression()
    model1.fit(X, y)
    evaluator1 = ModelEvaluator(model1)
    metrics1 = evaluator1.evaluate(X, y)
    print(metrics1)
    from sklearn.linear_model import LogisticRegression 
    sklearn_model1 = LogisticRegression()
    sklearn_model1.fit(X, y)
    sklearn_y_pred1 = sklearn_model1.predict(X)

    # Calculating metrics for scikit-learn Linear Regression
    sklearn_metrics1 = {
        'mse': np.mean((y - sklearn_y_pred1) ** 2)
    }
    print("\nScikit-learn Logistic Regression Metrics:")
    print(sklearn_metrics1)
    # Expected output: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0, 'mse': 0.0}
    model2 = KNNClassifier()
    model2.fit(X, y)
    evaluator2 = ModelEvaluator(model2)
    metrics2 = evaluator2.evaluate(X, y)
    print(metrics2)
    #import knn from sklearn
    from sklearn.neighbors import KNeighborsClassifier
    sklearn_model2 = KNeighborsClassifier()
    sklearn_model2.fit(X, y)
    sklearn_y_pred2 = sklearn_model2.predict(X)
    #metric
    sklearn_metrics2 = {
        'mse': np.mean((y - sklearn_y_pred2) ** 2)
    }
    print("\nScikit-learn KNN Metrics:")
    print(sklearn_metrics2)





    model3 = GaussianNB()
    model3.fit(X, y)
    evaluator3 = ModelEvaluator(model3)
    metrics3 = evaluator3.evaluate(X, y)
    print(metrics3)
    #import GaussianNB from sklearn
    from sklearn.naive_bayes import GaussianNB
    sklearn_model3 = GaussianNB()
    sklearn_model3.fit(X, y)
    sklearn_y_pred3 = sklearn_model3.predict(X)
    #metric
    sklearn_metrics3 = {
        'mse': np.mean((y - sklearn_y_pred3) ** 2)
    }
    print("\nScikit-learn GaussianNB Metrics:")
    print(sklearn_metrics3)



    model4 = DecisionTree()
    model4.fit(X, y)
    evaluator4 = ModelEvaluator(model4)
    metrics4 = evaluator4.evaluate(X, y)
    print(metrics4)
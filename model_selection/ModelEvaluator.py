import numpy as np
from supervised_learning.LinearRegression import LinearRegression
from supervised_learning.LogisticRegression import LogisticRegression

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
          
    # Expected output: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0, 'mse': 0.0}
    # Testing ModelEvaluator with Logistic Regression
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    model1 = LogisticRegression()
    model.fit(X, y)
    evaluator1 = ModelEvaluator(model1)
    metrics1 = evaluator1.evaluate(X, y)
    print(metrics1)
    # Expected output: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0, 'mse': 0.0}
    
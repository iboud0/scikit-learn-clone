# import numpy as np
# import sys
# sys.path.append('C:\\Users\\Pc\\Downloads\\scikit-learn-clone')

# from supervised_learning.LinearRegression import LinearRegression
# from supervised_learning.LogisticRegression import LogisticRegression
# from supervised_learning.knn import KNNClassifier
# from supervised_learning.NaiveBayes import GaussianNB
# from supervised_learning.decisiontrees import DecisionTreeRegressor
# from supervised_learning.decisiontrees import DecisionTreeClassifier
# from supervised_learning.decisiontrees import DecisionTree
# from supervised_learning.randomForest import RandomForestClassifier
# from supervised_learning.randomForest import RandomForestRegressor


# import numpy as np

# class ModelEvaluator:
#     def __init__(self, model):
#         self.model = model

#     def accuracy(self, y_true, y_pred):
#         """
#         Calculate the accuracy of the model.
#         """
#         if len(y_true) == 0:
#             raise ValueError("y_true is empty")
#         return np.sum(y_true == y_pred) / len(y_true)

#     def precision(self, y_true, y_pred):
#         """
#         Calculate the precision of the model.
#         """
#         tp = np.sum((y_true == 1) & (y_pred == 1))
#         fp = np.sum((y_true == 0) & (y_pred == 1))
#         return tp / (tp + fp) if (tp + fp) > 0 else 0

#     def recall(self, y_true, y_pred):
#         """
#         Calculate the recall of the model.
#         """
#         tp = np.sum((y_true == 1) & (y_pred == 1))
#         fn = np.sum((y_true == 1) & (y_pred == 0))
#         return tp / (tp + fn) if (tp + fn) > 0 else 0

#     def f1_score(self, y_true, y_pred):
#         """
#         Calculate the F1 score of the model.
#         """
#         prec = self.precision(y_true, y_pred)
#         rec = self.recall(y_true, y_pred)
#         return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

#     def roc_auc_score(self, y_true, y_pred):
#         """
#         Calculate the ROC-AUC score of the model.
#         """
#         # This implementation assumes binary classification
#         n = len(y_true)
#         tp = np.sum((y_true == 1) & (y_pred == 1))
#         fn = np.sum((y_true == 1) & (y_pred == 0))
#         fp = np.sum((y_true == 0) & (y_pred == 1))
#         tn = np.sum((y_true == 0) & (y_pred == 0))

#         tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
#         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

#         # Calculate the area under the ROC curve using the trapezoidal rule
#         tprs = [0] + [tpr] + [1]
#         fprs = [0] + [fpr] + [1]
#         area = np.sum([(fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2 for i in range(len(tprs) - 1)])

#         return area

#     def mean_squared_error(self, y_true, y_pred):
#         """
#         Calculate the mean squared error of the model.
#         """
#         return np.mean((y_true - y_pred) ** 2)

#     def evaluate(self, X, y):
#         """
#         Evaluate the model using various metrics.
#         """
#         y_pred = self.model.predict(X)
#         y_pred_cls = np.round(y_pred)

#         accuracy = self.accuracy(y, y_pred_cls)
#         precision = self.precision(y, y_pred_cls)
#         recall = self.recall(y, y_pred_cls)
#         f1 = self.f1_score(y, y_pred_cls)
#         roc_auc = self.roc_auc_score(y, y_pred)
#         mse = self.mean_squared_error(y, y_pred)

#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'roc_auc': roc_auc,
#             'mse': mse
#         }



# #test
# if __name__ == '__main__':
#     # Testing ModelEvaluator
#     X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
#     y = np.array([0, 0, 1, 1, 1])

#     # Linear Regression
#     model = LinearRegression()
#     model.fit(X, y)
#     evaluator = ModelEvaluator(model)
#     metrics = evaluator.evaluate(X, y)
#     print("\nCustom Linear Regression Metrics:")
#     print(metrics)

#     # Scikit-learn Linear Regression
#     from sklearn.linear_model import LinearRegression 
#     sklearn_model = LinearRegression()
#     sklearn_model.fit(X, y)
#     sklearn_y_pred = sklearn_model.predict(X)

#     sklearn_metrics = {
#         'mse': np.mean((y - sklearn_y_pred) ** 2),
#         'accuracy': np.sum(y == np.round(sklearn_y_pred)) / len(y),
#         'precision': np.sum((y == 1) & (np.round(sklearn_y_pred) == 1)) / np.sum(np.round(sklearn_y_pred) == 1),
#         'recall': np.sum((y == 1) & (np.round(sklearn_y_pred) == 1)) / np.sum(y == 1),
#         'f1': 2 * (np.sum((y == 1) & (np.round(sklearn_y_pred) == 1)) / np.sum(np.round(sklearn_y_pred) == 1) * np.sum((y == 1) & (np.round(sklearn_y_pred) == 1)) / np.sum(y == 1)) / (np.sum((y == 1) & (np.round(sklearn_y_pred) == 1)) / np.sum(np.round(sklearn_y_pred) == 1) + np.sum((y == 1) & (np.round(sklearn_y_pred) == 1)) / np.sum(y == 1)),
#         'roc_auc': 1  # ROC-AUC is not directly available for regression models
#     }
#     print("\nScikit-learn Linear Regression Metrics:")
#     print(sklearn_metrics)

#     # Logistic Regression
#     model1 = LogisticRegression()
#     model1.fit(X, y)
#     evaluator1 = ModelEvaluator(model1)
#     metrics1 = evaluator1.evaluate(X, y)
#     print("\nCustom Logistic Regression Metrics:")
#     print(metrics1)

#     # Scikit-learn Logistic Regression
#     from sklearn.linear_model import LogisticRegression 
#     sklearn_model1 = LogisticRegression()
#     sklearn_model1.fit(X, y)
#     sklearn_y_pred1 = sklearn_model1.predict(X)

#     sklearn_metrics1 = {
#         'mse': np.mean((y - sklearn_y_pred1) ** 2),
#         'accuracy': np.sum(y == sklearn_y_pred1) / len(y),
#         'precision': np.sum((y == 1) & (sklearn_y_pred1 == 1)) / np.sum(sklearn_y_pred1 == 1),
#         'recall': np.sum((y == 1) & (sklearn_y_pred1 == 1)) / np.sum(y == 1),
#         'f1': 2 * (np.sum((y == 1) & (sklearn_y_pred1 == 1)) / np.sum(sklearn_y_pred1 == 1) * np.sum((y == 1) & (sklearn_y_pred1 == 1)) / np.sum(y == 1)) / (np.sum((y == 1) & (sklearn_y_pred1 == 1)) / np.sum(sklearn_y_pred1 == 1) + np.sum((y == 1) & (sklearn_y_pred1 == 1)) / np.sum(y == 1)),
#         'roc_auc': 1  # ROC-AUC is not directly available for binary classification
#     }
#     print("\nScikit-learn Logistic Regression Metrics:")
#     print(sklearn_metrics1)

#     # KNN
#     model2 = KNNClassifier()
#     model2.fit(X, y)
#     evaluator2 = ModelEvaluator(model2)
#     metrics2 = evaluator2.evaluate(X, y)
#     print("\nCustom KNN Metrics:")
#     print(metrics2)

#     # Scikit-learn KNN
#     from sklearn.neighbors import KNeighborsClassifier
#     sklearn_model2 = KNeighborsClassifier()
#     sklearn_model2.fit(X, y)
#     sklearn_y_pred2 = sklearn_model2.predict(X)

#     sklearn_metrics2 = {
#         'mse': np.mean((y - sklearn_y_pred2) ** 2),
#         'accuracy': np.sum(y == sklearn_y_pred2) / len(y),
#         'precision': np.sum((y == 1) & (sklearn_y_pred2 == 1)) / np.sum(sklearn_y_pred2 == 1),
#         'recall': np.sum((y == 1) & (sklearn_y_pred2 == 1)) / np.sum(y == 1),
#         'f1': 2 * (np.sum((y == 1) & (sklearn_y_pred2 == 1)) / np.sum(sklearn_y_pred2 == 1) * np.sum((y == 1) & (sklearn_y_pred2 == 1)) / np.sum(y == 1)) / (np.sum((y == 1) & (sklearn_y_pred2 == 1)) / np.sum(sklearn_y_pred2 == 1) + np.sum((y == 1) & (sklearn_y_pred2 == 1)) / np.sum(y == 1)),
#         'roc_auc': 1  # ROC-AUC is not directly available for binary classification
#     }
#     print("\nScikit-learn KNN Metrics:")
#     print(sklearn_metrics2)

#     # GaussianNB
#     model3 = GaussianNB()
#     model3.fit(X, y)
#     evaluator3 = ModelEvaluator(model3)
#     metrics3 = evaluator3.evaluate(X, y)
#     print("\nCustom GaussianNB Metrics:")
#     print(metrics3)

#     # Scikit-learn GaussianNB
#     from sklearn.naive_bayes import GaussianNB
#     sklearn_model3 = GaussianNB()
#     sklearn_model3.fit(X, y)
#     sklearn_y_pred3 = sklearn_model3.predict(X)

#     sklearn_metrics3 = {
#         'mse': np.mean((y - sklearn_y_pred3) ** 2),
#         'accuracy': np.sum(y == sklearn_y_pred3) / len(y),
#         'precision': np.sum((y == 1) & (sklearn_y_pred3 == 1)) / np.sum(sklearn_y_pred3 == 1),
#         'recall': np.sum((y == 1) & (sklearn_y_pred3 == 1)) / np.sum(y == 1),
#         'f1': 2 * (np.sum((y == 1) & (sklearn_y_pred3 == 1)) / np.sum(sklearn_y_pred3 == 1) * np.sum((y == 1) & (sklearn_y_pred3 == 1)) / np.sum(y == 1)) / (np.sum((y == 1) & (sklearn_y_pred3 == 1)) / np.sum(sklearn_y_pred3 == 1) + np.sum((y == 1) & (sklearn_y_pred3 == 1)) / np.sum(y == 1)),
#         'roc_auc': 1  # ROC-AUC is not directly available for binary classification
#     }
#     print("\nScikit-learn GaussianNB Metrics:")
#     print(sklearn_metrics3)

#     # Decision Tree Regressor
#     from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

#     # Evaluate custom Decision Tree Regressor
#     model_reg = DecisionTreeRegressor()
#     model_reg.fit(X, y)
#     evaluator_reg = ModelEvaluator(model_reg)
#     metrics_reg = evaluator_reg.evaluate(X, y)
#     print("\nCustom Decision Tree Regressor Metrics:")
#     print(metrics_reg)

#     # Evaluate scikit-learn Decision Tree Regressor
#     sklearn_model_reg = SklearnDecisionTreeRegressor()
#     sklearn_model_reg.fit(X, y)
#     sklearn_y_pred_reg = sklearn_model_reg.predict(X)

#     sklearn_metrics_reg = {
#         'mse': np.mean((y - sklearn_y_pred_reg) ** 2),
#         # Other metrics are not applicable for regression
#     }
#     print("\nScikit-learn Decision Tree Regressor Metrics:")
#     print(sklearn_metrics_reg)

#     # Decision Tree Classifier
#     from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

#     # Evaluate custom Decision Tree Classifier
#     model_clf = DecisionTreeClassifier()
#     model_clf.fit(X, y)
#     evaluator_clf = ModelEvaluator(model_clf)
#     metrics_clf = evaluator_clf.evaluate(X, y)
#     print("\nCustom Decision Tree Classifier Metrics:")
#     print(metrics_clf)

#     # Evaluate scikit-learn Decision Tree Classifier
#     sklearn_model_clf = SklearnDecisionTreeClassifier()
#     sklearn_model_clf.fit(X, y)
#     sklearn_y_pred_clf = sklearn_model_clf.predict(X)

#     sklearn_metrics_clf = {
#         'accuracy': np.sum(y == sklearn_y_pred_clf) / len(y),
#         'precision': np.sum((y == 1) & (sklearn_y_pred_clf == 1)) / np.sum(sklearn_y_pred_clf == 1),
#         'recall': np.sum((y == 1) & (sklearn_y_pred_clf == 1)) / np.sum(y == 1),
#         'f1': 2 * (np.sum((y == 1) & (sklearn_y_pred_clf == 1)) / np.sum(sklearn_y_pred_clf == 1) * np.sum((y == 1) & (sklearn_y_pred_clf == 1)) / np.sum(y == 1)) / (np.sum((y == 1) & (sklearn_y_pred_clf == 1)) / np.sum(sklearn_y_pred_clf == 1) + np.sum((y == 1) & (sklearn_y_pred_clf == 1)) / np.sum(y == 1)),
#         'roc_auc': 1  # ROC-AUC is not directly available for binary classification
#     }
#     print("\nScikit-learn Decision Tree Classifier Metrics:")
#     print(sklearn_metrics_clf)
#     # Decision Tree
#     model4 = DecisionTree()
#     model4.fit(X, y)
#     evaluator4 = ModelEvaluator(model4)
#     metrics4 = evaluator4.evaluate(X, y)
#     print("\nCustom Decision Tree Metrics:")
#     print(metrics4)
#     # Scikit-learn Decision Tree
#     from sklearn.tree import DecisionTreeClassifier
#     sklearn_model4 = DecisionTreeClassifier()
#     sklearn_model4.fit(X, y)
#     sklearn_y_pred4 = sklearn_model4.predict(X)

#     sklearn_metrics4 = {
#         'mse': np.mean((y - sklearn_y_pred4) ** 2),
#         'accuracy': np.sum(y == sklearn_y_pred4) / len(y),
#         'precision': np.sum((y == 1) & (sklearn_y_pred4 == 1)) / np.sum(sklearn_y_pred4 == 1),
#         'recall': np.sum((y == 1) & (sklearn_y_pred4 == 1)) / np.sum(y == 1),
#         'f1': 2 * (np.sum((y == 1) & (sklearn_y_pred4 == 1)) / np.sum(sklearn_y_pred4 == 1) * np.sum((y == 1) & (sklearn_y_pred4 == 1)) / np.sum(y == 1)) / (np.sum((y == 1) & (sklearn_y_pred4 == 1)) / np.sum(sklearn_y_pred4 == 1) + np.sum((y == 1) & (sklearn_y_pred4 == 1)) / np.sum(y == 1)),
#         'roc_auc': 1  # ROC-AUC is not directly available for binary classification
#     }
#     print("\nScikit-learn Decision Tree Metrics:")
#     print(sklearn_metrics4)


#     # Testing RandomForestClassifier
#     X_cls = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
#     y_cls = np.array([0, 0, 1, 1, 1])

#     # Custom RandomForestClassifier
#     rf_cls = RandomForestClassifier(n_estimators=100)
#     rf_cls.fit(X_cls, y_cls)
#     evaluator_cls = ModelEvaluator(rf_cls)
#     metrics_cls = evaluator_cls.evaluate(X_cls, y_cls)
#     print("\nCustom RandomForestClassifier Metrics:")
#     print(metrics_cls)

#     # Scikit-learn RandomForestClassifier
#     from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
#     sk_rf_cls = SKRandomForestClassifier(n_estimators=100)
#     sk_rf_cls.fit(X_cls, y_cls)
#     sk_y_pred_cls = sk_rf_cls.predict(X_cls)
#     sk_metrics_cls = ModelEvaluator(sk_rf_cls).evaluate(X_cls, y_cls)
#     print("\nScikit-learn RandomForestClassifier Metrics:")
#     print(sk_metrics_cls)

#     # # Testing RandomForestRegressor
#     # X_reg = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
#     # y_reg = np.array([2, 3, 4, 5, 6])

#     # # Custom RandomForestRegressor
#     # rf_reg = RandomForestRegressor(n_estimators=100)
#     # rf_reg.fit(X_reg, y_reg)
#     # evaluator_reg = ModelEvaluator(rf_reg)
#     # metrics_reg = evaluator_reg.evaluate(X_reg, y_reg)
#     # print("\nCustom RandomForestRegressor Metrics:")
#     # print(metrics_reg)

#     # # Scikit-learn RandomForestRegressor
#     # from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
#     # sk_rf_reg = SKRandomForestRegressor(n_estimators=100)
#     # sk_rf_reg.fit(X_reg, y_reg)
#     # sk_y_pred_reg = sk_rf_reg.predict(X_reg)
#     # sk_metrics_reg = ModelEvaluator(sk_rf_reg).evaluate(X_reg, y_reg)
#     # print("\nScikit-learn RandomForestRegressor Metrics:")
#     # print(sk_metrics_reg)

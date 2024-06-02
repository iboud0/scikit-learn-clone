

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from sklearn.datasets import load_iris
from ensemble_methods.adaBoostClassifier import AdaBoostClassifier
from ensemble_methods.gradient_boosting import GradientBoostingClassifier
from ensemble_methods.gradient_boosting import GradientBoostingRegressor

from ensemble_methods.stacking import StackingClassifier
from ensemble_methods.stacking import StackingRegressor

from ensemble_methods.baggingClassifier import BaggingClassifier
import numpy as np
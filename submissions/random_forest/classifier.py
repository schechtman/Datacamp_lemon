

from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import RandomOverSampler

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline_imb(
            Imputer(strategy='median'),
            RandomOverSampler(),
            RandomForestClassifier(10, verbose=True, min_impurity_decrease=10e-5)
            )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
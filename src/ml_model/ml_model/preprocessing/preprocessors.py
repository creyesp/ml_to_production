import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare"
            )

        return X
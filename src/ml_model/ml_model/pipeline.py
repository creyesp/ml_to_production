from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone


from ml_model.preprocessing import preprocessors as pp
from ml_model.config import config

import logging


_logger = logging.getLogger(__name__)


numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(categories='auto', handle_unknown='ignore')),
    ])

preprocessor_num = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, config.NUMERICAL_FEATURES),
        ('passthrough', 'passthrough', config.BINARY_FEATURES),
        ('drop', 'drop', config.DROP_FEATURES + config.CATEGORICAL_FEATURES),
    ],
    n_jobs=-1)

preprocessor_cat = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, config.NUMERICAL_FEATURES),
        ('cat', categorical_transformer, config.CATEGORICAL_FEATURES),
        ('passthrough', 'passthrough', config.BINARY_FEATURES),
        ('drop', 'drop', config.DROP_FEATURES),
    ],
    n_jobs=-1)


rfp_num = Pipeline(
    [
        ('preprocessor', preprocessor_num),
        ('rf_regressor', RandomForestRegressor(n_estimators=100,
                                               min_samples_leaf=5,
                                               n_jobs=-1))
    ]
)

rfp = Pipeline(
    [
        ('preprocessor', preprocessor_cat),
        ('rf_regressor', RandomForestRegressor(n_estimators=100,
                                               min_samples_leaf=5,
                                               n_jobs=-1))
    ]
)

def get_model(name:str = 'rf'):
    if name == 'rf':
        model_ = clone(rfp)
    else:
        raise ValueError('invalid model name')

    return model_

class Model_Factory:
    def __init__(self, model_type):
        self.model_tyṕe = model_type

    @classmethod
    def from_name(cls):
        pass



from pathlib import Path
import os

import ml_model


PACKAGE_ROOT = Path(ml_model.__file__).resolve().parent

DATA_URL = "https://raw.githubusercontent.com/creyesp/Meetup_uy/master/data/ready/properties.csv"

TARGET = 'price'

DROP_FEATURES = [
    'year',
]
NUMERICAL_FEATURES = [
    'bathrooms',
    'bedrooms',
    'garage',
    'servide_fees',
    'surface_balcony',
    'surface_covered',
    'floor',
    'm2_index',
    'decade',
]
BINARY_FEATURES = [
    'floor_special',
    'facilities',
    'near_river',
    'is_house',
    'barbecue',
]
CATEGORICAL_FEATURES = [
    'zone',
    'orientation',
    'state',
]

FEATURES = \
    DROP_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

PIPELINE_NAME = "house_price_prediction"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')

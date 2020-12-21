from pathlib import Path

PACKAGE_ROOT = Path(ml_model.__file__).resolve().parent

DATA_URL = "https://github.com/creyesp/Meetup_uy/blob/master/data/ready/properties.csv"

TARGET = 'sale_price'

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

PIPELINE_NAME = "house_price_prediction"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

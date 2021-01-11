import logging

import numpy as np
from sklearn.model_selection import train_test_split

from ml_model import pipeline
from ml_model.data_management import load_dataset, save_pipeline
from ml_model.config import config
#from ml_model import __version__ as _version


_version = 'XXX'
_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.DATA_URL)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET],
        test_size=0.1,
        random_state=0
    )

    # transform the target
    #y_train = np.log(y_train)

    model = pipeline.get_model('rf').fit(X_train[config.FEATURES], y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(model=model)


if __name__ == "__main__":
    run_training()

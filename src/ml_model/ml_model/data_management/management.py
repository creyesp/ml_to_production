import os
import logging

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from ml_model.config import config
#from ml_model import __version__ as _version
_version = 'XXX'




_logger = logging.getLogger(__name__)


def load_dataset(file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(file_name)
    return _data


def save_pipeline(model, folder=None) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = os.path.join(config.TRAINED_MODEL_DIR, save_file_name)

    #remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(model, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
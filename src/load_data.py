import pandas as pd
from src.utils.logger import get_logger

logger = get_logger()


def load_data(path):

    logger.info("Dataset loading boshlandi")

    df = pd.read_csv(path)

    logger.info(f"Dataset shape: {df.shape}")

    return df
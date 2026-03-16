import re
from src.utils.logger import get_logger

logger = get_logger()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text


def preprocess_dataset(df):

    logger.info("Preprocessing boshlandi")

    df["text"] = df["text"].apply(clean_text)

    logger.info("Preprocessing tugadi")

    return df
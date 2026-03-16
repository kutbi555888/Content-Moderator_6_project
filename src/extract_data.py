import pandas as pd
from src.utils.logger import get_logger

logger = get_logger()


def extract_data(input_path, output_path):

    logger.info("Data extraction boshlandi")

    df = pd.read_csv(input_path)

    logger.info(f"Dataset yuklandi: {df.shape}")

    df.to_csv(output_path, index=False)

    logger.info("Data extraction tugadi")

    return df




from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.utils.logger import get_logger

logger = get_logger()


def create_tfidf(df):

    logger.info("TF-IDF feature extraction boshlandi")

    vectorizer = TfidfVectorizer(
        max_features=10000
    )

    X = vectorizer.fit_transform(df["text"])

    joblib.dump(vectorizer, "models/vectorizer_tfidf.joblib")

    logger.info("Vectorizer saqlandi")

    return X
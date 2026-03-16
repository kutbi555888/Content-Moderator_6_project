import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger


logger = get_logger()


def feature_importance_analysis(data_path, vectorizer_path, model_path):

    logger.info("Advanced EDA boshlandi")

    # -------------------------
    # DATA LOAD
    # -------------------------

    df = pd.read_csv(data_path)

    logger.info(f"Dataset yuklandi: {df.shape}")

    X = df["text"]
    y = df["label"]

    # -------------------------
    # TRAIN TEST SPLIT
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # LOAD VECTORiZER
    # -------------------------

    vectorizer = joblib.load(vectorizer_path)

    logger.info("TF-IDF vectorizer yuklandi")

    X_train_vec = vectorizer.transform(X_train)

    # -------------------------
    # LOAD MODEL
    # -------------------------

    model = joblib.load(model_path)

    logger.info("Model yuklandi")

    # -------------------------
    # FEATURE NAMES
    # -------------------------

    feature_names = vectorizer.get_feature_names_out()

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    importance = np.mean(np.abs(model.coef_), axis=0)

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    top_features = feature_importance.sort_values(
        by="importance",
        ascending=False
    ).head(20)

    logger.info("Top featurelar hisoblandi")

    # -------------------------
    # PLOT
    # -------------------------

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=top_features,
        x="importance",
        y="feature"
    )

    plt.title("Top 20 Important Features")

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.show()

    logger.info("Advanced EDA tugadi")

    return top_features
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack


def create_features(df):

    vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=5,
        max_df=0.9,
        ngram_range=(1,2),
        stop_words="english"
    )

    X_tfidf = vectorizer.fit_transform(df["clean_text"])


    stat_features = df[[
        "text_length"
    ]]

    scaler = MinMaxScaler()

    stat_scaled = scaler.fit_transform(stat_features)

    X_final = hstack([X_tfidf, stat_scaled])

    y = df["label"]

    joblib.dump(vectorizer, "models/vectorizer_engineered.joblib")
    joblib.dump(scaler, "models/stat_scaler.joblib")

    return X_final, y
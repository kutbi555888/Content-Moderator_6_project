import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_stopwords(text):

    words = text.split()

    words = [w for w in words if w not in ENGLISH_STOP_WORDS]

    return " ".join(words)


def preprocess_data(df):

    df["clean_text"] = df["text"].apply(clean_text)

    df["clean_text"] = df["clean_text"].apply(remove_stopwords)

    df = df.drop_duplicates(subset="clean_text")

    df["text_length"] = df["clean_text"].apply(len)

    df = df[df["text_length"] > 20]

    return df
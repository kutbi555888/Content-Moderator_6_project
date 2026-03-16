from src.data.load_data import load_data
from src.features.feature_extraction import create_tfidf

df = load_data("Data/processed/wiki_dataset.csv")

X = create_tfidf(df)
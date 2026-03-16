from src.data.load_data import load_data
from src.preprocessing.baseline_preprocess import preprocess_dataset

df = load_data("Data/processed/wiki_dataset.csv")

df = preprocess_dataset(df)
from src.data.load_data import load_data
from src.eda.baseline_eda import class_distribution

df = load_data("Data/processed/wiki_dataset.csv")

class_distribution(df)
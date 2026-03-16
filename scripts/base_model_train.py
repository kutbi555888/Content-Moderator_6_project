from src.data.load_data import load_data
from src.models.baseline_train import train_baseline

df = load_data("Data/processed/wiki_dataset.csv")

train_baseline(df)
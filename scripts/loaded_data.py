from src.data.load_data import load_data

df = load_data("Data/processed/wiki_dataset.csv")

print(df.head())
import pandas as pd
from src.preprocessing.improvement_preprocess import preprocess_data


df = pd.read_csv("Data/Raw_Data/Extract_data/wiki_dataset.csv")

df = preprocess_data(df)

df.to_csv("Data/Processed_Data/improved_dataset.csv", index=False)

print("Improved preprocessing completed")
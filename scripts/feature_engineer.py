import pandas as pd
import joblib
from src.features.feature_engineering import create_features


df = pd.read_csv("Data/Processed_Data/improved_dataset.csv")

X, y = create_features(df)

joblib.dump(X, "Data/Engineered_data/X_engineered.joblib")
joblib.dump(y, "Data/Engineered_data/y_engineered.joblib")

print("Feature engineering completed")
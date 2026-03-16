import joblib
from src.models.improvement_train import train_models


X = joblib.load("Data/Engineered_data/X_engineered.joblib")
y = joblib.load("Data/Engineered_data/y_engineered.joblib")

bag_lr, voting, stack = train_models(X, y)

joblib.dump(bag_lr, "models/bagging_model.joblib")
joblib.dump(voting, "models/voting_model.joblib")
joblib.dump(stack, "models/stacking_model.joblib")

print("Improvement models trained")
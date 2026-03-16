import joblib
from src.models.tuning import tune_model
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


X = joblib.load("Data/Engineered_data/X_engineered.joblib")
y = joblib.load("Data/Engineered_data/y_engineered.joblib")

model = joblib.load("models/stacking_model.joblib")

param_grid = {

    "final_estimator__C":[0.1,1,10]
}

best_model = tune_model(model, param_grid, X, y)

joblib.dump(best_model, "models/best_model.joblib")

print("Best tuned model saved")
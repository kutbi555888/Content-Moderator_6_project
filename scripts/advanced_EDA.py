from src.eda.advanced_eda import feature_importance_analysis

DATA_PATH = "Data/Raw_Data/Extract_data/wiki_dataset.csv"

VECTORIZER_PATH = "models/vectorizer_tfidf.joblib"

MODEL_PATH = "models/baseline_linear_svm.joblib"


feature_importance_analysis(
    DATA_PATH,
    VECTORIZER_PATH,
    MODEL_PATH
)
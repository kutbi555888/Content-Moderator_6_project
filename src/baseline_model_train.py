import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from src.utils.logger import get_logger

logger = get_logger()


def train_baseline(df):

    logger.info("Baseline model training boshlandi")

    X = df["text"]
    y = df["label"]

    vectorizer = joblib.load("models/vectorizer_tfidf.joblib")

    X_vec = vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearSVC()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    logger.info("Baseline training tugadi")

    joblib.dump(model, "models/baseline_svm.joblib")
    
    
    
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(n_estimators=200)
    svm = LinearSVC()

    bag_lr = BaggingClassifier(
        estimator=lr,
        n_estimators=10
    )

    voting = VotingClassifier(
        estimators=[
            ("nb", nb),
            ("lr", lr),
            ("rf", rf),
            ("svm", svm)
        ],
        voting="hard"
    )

    stack = StackingClassifier(
        estimators=[
            ("nb", nb),
            ("lr", lr),
            ("rf", rf),
            ("svm", svm)
        ],
        final_estimator=LogisticRegression()
    )

    bag_lr.fit(X_train, y_train)
    voting.fit(X_train, y_train)
    stack.fit(X_train, y_train)

    return bag_lr, voting, stack
import joblib
from sklearn.model_selection import RandomizedSearchCV


def tune_model(model, param_grid, X_train, y_train):

    search = RandomizedSearchCV(

        model,

        param_distributions=param_grid,

        n_iter=20,

        cv=5,

        n_jobs=-1,

        random_state=42,

        verbose=1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_
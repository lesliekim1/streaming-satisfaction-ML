import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

# Purpose: Use random forest pipeline to do grid search.
# Input: none
# Output: best hyperparameter result
def rf_train():
    pipe_rf = Pipeline([
        ('rf', RandomForestClassifier(random_state=1))
    ])

    rf_param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10]
    }

    search_rf = GridSearchCV(estimator=pipe_rf,
                             param_grid=rf_param_grid,
                             scoring='f1_weighted',
                             cv=2,
                             n_jobs=6,
                             refit=True,
                             verbose=2)

    return search_rf

# Purpose: use logistic regression pipeline to do grid search.
# Input: none
# Output: best hyperparameter result
def log_train():
    pipe_log = Pipeline([
        ('ss', StandardScaler()),
        ('log', LogisticRegression())
    ])

    param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    log_param_grid = [{
        'log__C': param_range,
        'log__penalty': ['l2'],
        'log__solver': ['lbfgs'],
        'log__max_iter': [1000]
    }]

    search_log = GridSearchCV(estimator=pipe_log,
                              param_grid=log_param_grid,
                              scoring='f1_weighted',
                              cv=2,
                              n_jobs=6,
                              refit=True,
                              verbose=2)

    return search_log

# Purpose: Output best model's f1-score and hyperparameters for each algorithm.
# Input: search_rf, search_log
# Output: print best f1-scores and parameters
def output_best_score_and_params(search_rf, search_log):
    print(f"Random Forest best f1-score: {search_rf.best_score_}")
    print(f"Random Forest best params: {search_rf.best_params_}")
    print()
    print(f"Logistic Regression best f1-score: {search_log.best_score_}")
    print(f"Logistic Regression best params: {search_log.best_params_}")

# Purpose: Compare two models and pick the best one based on f1-score.
# Input: search_rf, search_log
# Output: best_model
def find_best_model(search_rf, search_log):
    if search_rf.best_score_ > search_log.best_score_:
        best_model = search_rf.best_estimator_
        best_model_name = "Random Forest"
        best_params = search_rf.best_params_
    else:
        best_model = search_log.best_estimator_
        best_model_name = "Logistic Regression"
        best_params = search_log.best_params_

    print(f"\nFinal Model: {best_model_name}")
    print(f"Best hyperparameters: {best_params}")

    return best_model

# Purpose: Output the confusion matrix and classification report of the final model.
# Input: final model, test features, test labels
# Output: confusion matrix and classification report
def test_final_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("\nConfusion Matrix:")
    print(confmat)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

# Purpose: Run the program to do analysis on streaming satisfaction survey.
# Input: none 
# Output: none
def main():
    df = pd.read_csv("processed_stream.csv", skipinitialspace=True)

    target_col = "How likely are you to recommend Netflix to someone looking for a streaming service, knowing that they have recently cracked down on password sharing?"
    targets = df[target_col]
    data = df.drop([target_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets, test_size=0.2, random_state=42)

    # Run random forest pipeline and grid search
    search_rf = rf_train()
    search_rf.fit(X_train, y_train)

    # Run logistic regression pipeline and grid search
    search_log = log_train()
    search_log.fit(X_train, y_train)

    print("\n----------")
    output_best_score_and_params(search_rf, search_log)
    print("----------")

    # Determine best model
    best_model = find_best_model(search_rf, search_log)

    # Test final model
    test_final_model(best_model, X_test, y_test)

main()

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

    searchRF = GridSearchCV(estimator=pipe_rf,
                            param_grid=rf_param_grid,
                            scoring='f1_weighted',
                            cv=2,
                            n_jobs=6,
                            refit=True,
                            verbose=2)

    return searchRF

# Purpose: use logisitic regression pipeline to do grid search.
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

    searchLR = GridSearchCV(estimator=pipe_log,
                            param_grid=log_param_grid,
                            scoring='f1_weighted',
                            cv=2,
                            n_jobs=6,
                            refit=True,
                            verbose=2)

    return searchLR

# Purpose: Output the confusion matrix and classfication report of the final model.
# Input: final model, test features, test labels
# Output: confusion matrix and classification report
def test_final_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix:")
    print(confmat)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

# Purpose: Run the program.
# Input: none 
# Output: none
def main():
    df = pd.read_csv("processed_stream.csv", skipinitialspace=True)

    target_col = "How likely are you to recommend Netflix to someone looking for a streaming service, knowing that they have recently cracked down on password sharing?"
    targets = df[target_col]
    data = df.drop([target_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets, test_size=0.2, random_state=42)

    # Run first pipeline on random forest
    searchRF = rf_train()
    searchRF.fit(X_train, y_train)

    print("\n----------")
    print(f"Random Forest best f1-score: {searchRF.best_score_}")
    print(f"Random Forest best params: {searchRF.best_params_}")
    print("----------")

    test_final_model(searchRF.best_estimator_, X_test, y_test)

    print()
    # Run second pipeline on logisitic regression
    searchLOG = log_train()
    searchLOG.fit(X_train, y_train)

    print("\n----------")
    print(f"Logisitic Regression best f1-score: {searchLOG.best_score_}")
    print(f"Logistic Regression best params: {searchLOG.best_params_}")
    print("----------")

    test_final_model(searchLOG.best_estimator_, X_test, y_test)

main()


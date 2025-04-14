# Streaming Satisfaction ML

A personal machine learning project to analyze customer satisfaction on Netflix using a dataset I collected.

## Project Overview
This project involves:
- **Data preprocessing** 
- **Train-test splitting** to evaluate model performance
- **Pipeline setup** to streamline transformations and modeling
- **Model training** using:
  - Random Forest Classifier
  - Logistic Regression
- **Hyperparameter tuning** using GridSearchCV for optimization and model selection

## Hyperparameter Tuning
To optimize model performance, I used **GridSearchCV** to systematically explore different combinations of hyperparameters for both Random Forest and Logistic Regression, selecting the best-performing configuration based on cross-validated F1 scores.

## Results

Due to the small size of the dataset and the uneven distribution across the five satisfaction levels (classes), no strong conclusions could be drawn from the results. The models ended up struggling because some classes had very few examples, making predictions unstable and metrics unreliable. Therefore, this project ended up being a practice in using Grid Search and getting familiar with the models used. Next time, more data should be collected or oversampling should've been done. 

**output:**
```
Random Forest best f1-score: 0.36991341991341986
Random Forest best params: {'rf__max_depth': None, 'rf__min_samples_split': 2, 'rf__n_estimators': 200}
```
```
Confusion Matrix:
[[1 1 0 0 0]
 [0 0 2 0 0]
 [0 0 0 0 1]
 [1 0 0 0 1]
 [0 1 0 0 0]]
Classification Report:
              precision    recall  f1-score   support

         1.0       0.50      0.50      0.50         2
         2.0       0.00      0.00      0.00         2
         3.0       0.00      0.00      0.00         1
         4.0       0.00      0.00      0.00         2
         5.0       0.00      0.00      0.00         1

    accuracy                           0.12         8
   macro avg       0.10      0.10      0.10         8
weighted avg       0.12      0.12      0.12         8
```

## Note
The dataset used in this project is private due to confidentiality reasons and will not be publicly shared.

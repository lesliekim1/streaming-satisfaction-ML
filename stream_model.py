import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import numpy as np


def main():
    df = pd.read_csv("processed_stream.csv", skipinitialspace=True)
    
    # Get train and testing data
    targets = df["How likely are you to recommend Netflix to someone looking for a streaming service, knowing that they have recently cracked down on password sharing?"] 
    data = df.drop(["How likely are you to recommend Netflix to someone looking for a streaming service, knowing that they have recently cracked down on password sharing?"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets, test_size=0.2)

    searchLDA = lda_train()
    searchLDA = searchLDA.fit(X_train, y_train)

main()

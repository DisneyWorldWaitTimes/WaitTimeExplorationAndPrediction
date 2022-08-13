import joblib
import os
import json

import pandas as pd
from sklearn import metrics
from pipeline_train import impute_transform
import importlib.util



def setup():
    with open("data/processed/dtypes_parsed.json") as json_file:
        dtypes = json.load(json_file)

    return dtypes


def predict_and_get_metrics(modelPkl, X_test, y_test):
    model = joblib.load(modelPkl)
    predictions = model.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    regression_metrics = {"Mean Absolute Error (MAE)": mae,
                          "Mean Squared Error (MSE)": mse,
                          "R-Squared": r2}

    print(regression_metrics)
    return predictions, regression_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Pipeline Pickle (.pkl.gz)")
    args = parser.parse_args()

    with open("src/models/dtypes.json") as json_file:
        dtypes_final = json.load(json_file)

    X_test = pd.read_csv("data/final/X_test_posted_final.csv", dtype=dtypes_final, compression='gzip')
    X_test = X_test.drop(columns=['Unnamed: 0'])
    y_test = pd.read_csv("data/final/y_test_posted_final.csv", dtype=dtypes_final, compression='gzip')
    y_test = y_test["POSTED_WAIT"]

    preds, regression_metrics = predict_and_get_metrics(args.input, X_test, y_test)


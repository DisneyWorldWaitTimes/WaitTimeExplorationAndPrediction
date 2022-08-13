import joblib
import json

import pandas as pd
from sklearn import metrics
from pipeline_train import impute_transform


def setup():
    with open("data/processed/dtypes_parsed.json") as json_file:
        dtypes = json.load(json_file)

    return dtypes


def predict_and_get_metrics(modelPkl, X_test, y_test):
    """
        Predict wait times for test datasets based on pickled model from pipeline_train.py

        Parameters
        ----------
        modelPkl: String
            Pickle file path from pipeline_train.py
        X_test: DataFrame
            Clean feature DataFrame ready for pipeline predictions
        y_test: Series
            Clean targets list ready for metrics calculation

        Returns
        -------
        predictions: Series
            predictions for each X_test entry
        regression_metrics: Dictionary
            Metrics dictionary with key performance metrics

    """
    # load model from pkl
    model = joblib.load(modelPkl)

    # make predictions based on test features
    predictions = model.predict(X_test)

    # calculate performance metrics
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    # generate regression metrics dictionary
    regression_metrics = {"Mean Absolute Error (MAE)": mae,
                          "Mean Squared Error (MSE)": mse,
                          "R-Squared": r2}

    print(regression_metrics)
    return predictions, regression_metrics


if __name__ == '__main__':
    import argparse

    # parse input and output args
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Pipeline Pickle (.pkl.gz)")
    args = parser.parse_args()

    # load final data types
    with open("src/models/dtypes.json") as json_file:
        dtypes_final = json.load(json_file)

    # Load testing data
    X_test = pd.read_csv("data/final/X_test_posted_final.csv", dtype=dtypes_final, compression='gzip')
    X_test = X_test.drop(columns=['Unnamed: 0'])
    y_test = pd.read_csv("data/final/y_test_posted_final.csv", dtype=dtypes_final, compression='gzip')
    y_test = y_test["POSTED_WAIT"]

    preds, regression_metrics = predict_and_get_metrics(args.input, X_test, y_test)

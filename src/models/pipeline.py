import numpy as np
import pandas as pd
import os
import joblib
import swifter

from feature_engineering import parse_times
from sklearn.compose import make_column_selector as selector, make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xg


from scipy.stats import skew
import json

def setup():
    with open("data/processed/dtypes_parsed.json") as json_file:
        dtypes = json.load(json_file)

    return dtypes

def impute_transform(x):
    for col in x:
        if col in parse_times:
            x[col] = x[col].fillna("99")
            x[col] = x[col].apply(lambda h: h[:2] if h[0] != 0 else h[:1]).astype(int).astype("Int8")

        x[col] = x[col].fillna(method='bfill')
        x[col] = x[col].fillna(x[col].median())

        if (x[col].dtype != "bool") and (abs(skew(list(x[col]))) > 0.8):
            # +20 linear scale on all values to ensure no resulting -inf vals
            x[f"log_{col}"] = x[col].swifter.apply(lambda k: np.log(k + 20))

            x.drop(columns=[col], inplace=True)

    return x


def pipeline_train(X_train, y_train):
    preprocessor = make_column_transformer(
        (VarianceThreshold(threshold=0.001), selector(dtype_include="bool")),
        (RobustScaler(), selector(dtype_include=np.number)), remainder='passthrough')

    pipeline = Pipeline(
        steps=[("imputerAndLogTransformer", FunctionTransformer(impute_transform)),
               ("preprocessor", preprocessor),
               ("regressor", RandomForestRegressor(n_estimators=10, max_depth=50, n_jobs=-1, random_state=0))]
    )

    pipeline.fit(X_train, y_train)

    return pipeline

def predict_and_get_metrics(model, X_test, y_test):
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
    dtypes = setup()

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Final Data Folder (output of feature_engineering.py)")
    parser.add_argument('output', help="Pipeline Pickle (.pkl)")
    args = parser.parse_args()

    for file in os.listdir(args.input):
        if "X_train" in str(file):
            X_train = pd.read_csv(f"{args.input}/{file}", dtype=dtypes, compression='gzip')
            X_train = X_train.drop(columns=['Unnamed: 0'])
            print(X_train.iloc[0])
        elif "y_train" in str(file):
            y_train = pd.read_csv(f"{args.input}/{file}", dtype=dtypes, compression='gzip')
        elif "X_test" in str(file):
            X_test = pd.read_csv(f"{args.input}/{file}", dtype=dtypes, compression='gzip')
        elif "y_test" in str(file):
            y_test = pd.read_csv(f"{args.input}/{file}", dtype=dtypes, compression='gzip')


    allCols = list(X_train.columns)
    pipeline = pipeline_train(X_train, y_train)
    predictions, regression_metrics = predict_and_get_metrics(pipeline, X_test, y_test)

    joblib.dump(pipeline, f'{args.output}.gz', compress=('gzip', 5))

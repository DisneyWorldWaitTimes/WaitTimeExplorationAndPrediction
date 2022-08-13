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
import importlib.util

from scipy.stats import skew
import json


def setup():
    with open("data/processed/dtypes_parsed.json") as json_file:
        dtypes = json.load(json_file)

    return dtypes


def feature_engineering():
    # specify the module that needs to be
    # imported relative to the path of the
    # module
    spec = importlib.util.spec_from_file_location("loadTrainTestPostedWaitTimes", "src/data/loadTrainTestData.py")

    # creates a new module based on spec
    loadTrainPosted = importlib.util.module_from_spec(spec)

    # executes the module in its own namespace
    # when a module is imported or reloaded.
    spec.loader.exec_module(loadTrainPosted)

    X_train, X_test, y_train, y_test = loadTrainPosted.loadTrainTestPostedWaitTimes()

    X_train["MONTHOFYEAR"] = X_train["date"].dt.month.astype("Int8")
    X_train["YEAR"] = X_train["date"].dt.year.astype("Int16")
    X_train["DAYOFYEAR"] = X_train["date"].dt.dayofyear.astype("Int16")
    X_train["HOUROFDAY"] = X_train["datetime"].dt.hour.astype("Int8")

    X_test["MONTHOFYEAR"] = X_test["date"].dt.month.astype("Int8")
    X_test["YEAR"] = X_test["date"].dt.year.astype("Int16")
    X_test["DAYOFYEAR"] = X_test["date"].dt.dayofyear.astype("Int16")
    X_test["HOUROFDAY"] = X_test["datetime"].dt.hour.astype("Int8")

    train = pd.concat([X_train, y_train], axis=1).sort_values(['datetime'])
    test = pd.concat([X_test, y_test], axis=1).sort_values(['datetime'])

    X_train_impute = train.drop(columns=["POSTED_WAIT"])
    y_train = train["POSTED_WAIT"]

    X_test_impute = test.drop(columns=["POSTED_WAIT"])
    y_test = test["POSTED_WAIT"]

    X_train_clean = X_train_impute.drop(columns=['date', 'datetime', 'Unnamed: 0'])
    X_test_clean = X_test_impute.drop(columns=['date', 'datetime', 'Unnamed: 0'])

    return X_train_clean, X_test_clean, y_train, y_test


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

if __name__ == '__main__':
    import argparse

    dtypes = setup()

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Final Data Folder (output of feature_engineering.py)")
    parser.add_argument('output', help="Pipeline Pickle (.pkl)")
    args = parser.parse_args()

    with open("src/models/dtypes.json") as json_file:
        dtypes_final = json.load(json_file)

    X_train = pd.read_csv(f"{args.input}/X_train_posted_final.csv", dtype=dtypes_final, compression='gzip')
    X_train = X_train.drop(columns=['Unnamed: 0'])
    y_train = pd.read_csv(f"{args.input}/y_train_posted_final.csv", dtype=dtypes_final, compression='gzip')
    y_train = y_train["POSTED_WAIT"]

    X_test = pd.read_csv(f"{args.input}/X_test_posted_final.csv", dtype=dtypes_final, compression='gzip')
    X_test = X_test.drop(columns=['Unnamed: 0'])
    y_test = pd.read_csv(f"{args.input}/y_test_posted_final.csv", dtype=dtypes_final, compression='gzip')
    y_test = y_test["POSTED_WAIT"]

    allCols = list(X_train.columns)
    pipeline = pipeline_train(X_train, y_train)

    joblib.dump(pipeline, f'{args.output}.gz', compress=('gzip', 5))

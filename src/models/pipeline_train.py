import numpy as np
import pandas as pd
import joblib
import swifter

from feature_engineering import parse_times
from sklearn.compose import make_column_selector as selector, make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import skew
import json


def impute_transform(x):
    """
            Perform data imputation as necessary & log transformation to skewed numeric columns

            Parameters
            ----------
            x: DataFrame
                Clean DataFrame ready for data imputation & transformation

            Returns
            -------
            x: DataFrame
                DataFrame ready for next step in pipeline
        """
    # iterate through columns in dataframe
    for col in x:
        # convert HH:MM to integer hour, filling missing with 99 to differentiate
        if col in parse_times:
            x[col] = x[col].fillna("99")
            x[col] = x[col].apply(lambda h: h[:2] if h[0] != 0 else h[:1]).astype(int).astype("Int8")

        # backfill imputation to fill with similar days and any remaining fill with median
        x[col] = x[col].fillna(method='bfill')
        x[col] = x[col].fillna(x[col].median())

        # check skew in numeric columns and log transform if necessary
        if (x[col].dtype != "bool") and (abs(skew(list(x[col]))) > 0.8):
            # +20 linear scale on all values to ensure no resulting -inf vals
            x[f"log_{col}"] = x[col].swifter.apply(lambda k: np.log(k + 20))
            # drop old columns
            x.drop(columns=[col], inplace=True)

    return x


def pipeline_train(X_train, y_train):
    """
            Train the pipeline for final model

            Parameters
            ----------
            X_train: DataFrame
                Clean feature DataFrame ready for pipeline transformation & fitting
            y_train: Series
                Clean targets list ready for pipeline fitting

            Returns
            -------
            pipeline: sklearn Pipeline object
                Fitted model with transformed data
        """
    preprocessor = make_column_transformer(
        # (VarianceThreshold(threshold=0.001), selector(dtype_include="bool")),
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

    # parse input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Final Data Folder (output of feature_engineering.py)")
    parser.add_argument('output', help="Pipeline Pickle (.pkl)")
    args = parser.parse_args()

    # load data types that match final clean dataframes
    with open("src/models/dtypes.json") as json_file:
        dtypes_final = json.load(json_file)

    # import pandas dataframes
    X_train = pd.read_csv(f"{args.input}/X_train_posted_final.csv", dtype=dtypes_final, compression='gzip')
    X_train = X_train.drop(columns=['Unnamed: 0'])
    y_train = pd.read_csv(f"{args.input}/y_train_posted_final.csv", dtype=dtypes_final, compression='gzip')
    y_train = y_train["POSTED_WAIT"]

    pipeline = pipeline_train(X_train, y_train)

    # dump pipeline into compressed pickle file as defined in output argument
    joblib.dump(pipeline, f'{args.output}.gz', compress=('gzip', 5))

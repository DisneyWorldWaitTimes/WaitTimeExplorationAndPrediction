import pandas as pd
import json

parse_times = ["MKOPEN", "MKCLOSE", "MKEMHOPEN", "MKEMHCLOSE",
                   "MKOPENYEST", "MKCLOSEYEST", "MKOPENTOM",
                   "MKCLOSETOM", "EPOPEN", "EPCLOSE", "EPEMHOPEN",
                   "EPEMHCLOSE", "EPOPENYEST", "EPCLOSEYEST",
                   "EPOPENTOM", "EPCLOSETOM", "HSOPEN", "HSCLOSE",
                   "HSEMHOPEN", "HSEMHCLOSE", "HSOPENYEST", "HSCLOSEYEST",
                   "HSOPENTOM", "HSCLOSETOM", "AKOPEN", "AKCLOSE",
                   "AKEMHOPEN", "AKOPENYEST", "AKCLOSEYEST", "AKEMHCLOSE",
                   "AKOPENTOM", "AKCLOSETOM", "MKPRDDT1", "MKPRDDT2",
                   "MKPRDNT1", "MKPRDNT2", "MKFIRET1", "MKFIRET2",
                   "EPFIRET1", "EPFIRET2", "HSPRDDT1", "HSFIRET1",
                   "HSFIRET2", "HSSHWNT1", "HSSHWNT2", "AKPRDDT1",
                   "AKPRDDT2", "AKSHWNT1", "AKSHWNT2"]

def setup():
    with open("data/processed/dtypes_parsed.json") as json_file:
        dtypes = json.load(json_file)

    return dtypes


def load_train_test_posted_wait_times(input_dir):
    print("LOADING DATA")
    """
            Loads train test data for posted wait times

            How to use:

            import importlib.util

            spec = importlib.util.spec_from_file_location("loadTrainTestPostedWaitTimes", "src/data/loadTrainTestData.py")
            loadTrainPosted = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loadTrainPosted)

            X_train_posted, X_test_posted, y_train_posted, y_test_posted = loadTrainPosted.loadTrainTestPostedWaitTimes()

            Parameters
            ----------

            Returns
            -------
            rideDataDf_trainX - train data features for posted wait times
            rideDataDf_testX - test data features for posted wait times
            rideDataDf_trainY - train data targets for posted wait times
            rideDataDf_testY - test data targets for posted wait times

        """
    parse_dates = ['date', 'datetime']
    X_train_list = []
    y_train_list = []
    dtypes = setup()

    for year in range(2015, 2022):

        rideData = pd.read_csv(f'{input_dir}/All_train_postedtimes{year}.csv', dtype=dtypes,
                               parse_dates=parse_dates, compression='gzip')
        rideDataX = rideData.drop(columns=["POSTED_WAIT"])
        rideDataY = rideData["POSTED_WAIT"]
        X_train_list.append(rideDataX)
        y_train_list.append(rideDataY)

    rideDataDf_trainX = pd.concat(X_train_list, ignore_index=True)
    rideDataDf_trainY = pd.concat(y_train_list, ignore_index=True)


    X_test_list = []
    y_test_list = []

    for year in range(2015, 2022):
        rideData = pd.read_csv(f'{input_dir}/All_test_postedtimes{year}.csv', dtype=dtypes,
                               parse_dates=parse_dates, compression='gzip')
        rideDataX = rideData.drop(columns=["POSTED_WAIT"])
        rideDataY = rideData["POSTED_WAIT"]

        X_test_list.append(rideDataX)
        y_test_list.append(rideDataY)

    rideDataDf_testX = pd.concat(X_test_list, ignore_index=True)
    rideDataDf_testY = pd.concat(y_test_list, ignore_index=True)

    return rideDataDf_trainX, rideDataDf_testX, rideDataDf_trainY, rideDataDf_testY


def data_preparation_for_pipeline(X_train, X_test, y_train, y_test):
    print("STARTING DATA PREP")
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


def feature_engineering(input_dir):
    X_train, X_test, y_train, y_test = load_train_test_posted_wait_times(input_dir)
    X_train_clean, X_test_clean, y_train, y_test = data_preparation_for_pipeline(X_train, X_test, y_train, y_test)

    print(dict(X_train_clean.dtypes))
    return X_train_clean, X_test_clean, y_train, y_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Clean Train/Test Data")
    parser.add_argument('output', help="Engineered data directory")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_train_test_posted_wait_times(args.input)
    X_train_clean, X_test_clean, y_train, y_test = data_preparation_for_pipeline(X_train, X_test, y_train, y_test)
    print(dict(X_train_clean.dtypes))
    print("WRITING FILES")
    for idx, file in enumerate([X_train_clean, X_test_clean, y_train, y_test]):
        print(file.shape)
        names = ["X_train", "X_test", "y_train", "y_test"]
        file.to_csv(f"{args.output}/{names[idx]}_posted_final.csv", compression='gzip')

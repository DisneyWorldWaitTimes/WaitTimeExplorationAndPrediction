import json

import pandas as pd

bool_dtypes = [
    "HOLIDAY", "WDWevent", "WDWrace", "MKevent", "EPevent", "HSevent",
    "AKevent", "MKEMHMORN", "MKEMHMYEST", "MKEMHMTOM", "MKEMHEVE",
    "MKEMHEYEST","MKEMHETOM", "EPEMHMORN", "EPEMHMYEST","EPEMHMTOM",
    "EPEMHEVE", "EPEMHEYEST", "EPEMHETOM",   "HSEMHMORN", "HSEMHMYEST",
    "HSEMHMTOM", "HSEMHEVE", "HSEMHEYEST", "HSEMHETOM",  "AKEMHMORN",
    "AKEMHMYEST", "AKEMHMTOM", "AKEMHEVE", "AKEMHEYEST", "AKEMHETOM"
]

with open("data/processed/dtypes_parsed.json") as json_file:
    dtypes = json.load(json_file)


def loadTrainTestPostedWaitTimes():
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

    for year in range(2015, 2022):

        rideData = pd.read_csv(f'data/processed/All_train_postedtimes{year}.csv', dtype=dtypes,
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
        rideData = pd.read_csv(f'data/processed/All_test_postedtimes{year}.csv', dtype=dtypes,
                               parse_dates=parse_dates, compression='gzip')
        rideDataX = rideData.drop(columns=["POSTED_WAIT"])
        rideDataY = rideData["POSTED_WAIT"]

        X_test_list.append(rideDataX)
        y_test_list.append(rideDataY)

    rideDataDf_testX = pd.concat(X_train_list, ignore_index=True)
    rideDataDf_testY = pd.concat(y_train_list, ignore_index=True)

    return rideDataDf_trainX, rideDataDf_testX, rideDataDf_trainY, rideDataDf_testY


def loadTrainTestActualWaitTimes():
    """
            Loads train test data for actual wait times

            How to use:

            import importlib.util

            spec = importlib.util.spec_from_file_location("loadTrainTestActualWaitTimes", "src/data/loadTrainTestData.py")
            loadTrainActual = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loadTrainActual)

            X_train_actual, X_test_actual, y_train_actual, y_test_actual = loadTrainPosted.loadTrainTestActualWaitTimes()

            Parameters
            ----------

            Returns
            -------
            rideDataDf_trainX - train data features for actual wait times
            rideDataDf_testX - test data features for actual wait times
            rideDataDf_trainY - train data targets for actual wait times
            rideDataDf_testY - test data targets for actual wait times

        """
    parse_dates = ['date', 'datetime']

    rideDataDf_trainX = pd.read_csv(f'data/processed/Xtrain_actualtimes.csv', dtype=dtypes,
                                parse_dates=parse_dates, compression='gzip')

    rideDataDf_trainY = pd.read_csv(f'data/processed/ytrain_actualtimes.csv', dtype=dtypes, compression='gzip')

    rideDataDf_testX = pd.read_csv(f'data/processed/Xtest_actualtimes.csv', dtype=dtypes,
                                parse_dates=parse_dates, compression='gzip')
    rideDataDf_testY = pd.read_csv(f'data/processed/ytrain_actualtimes.csv', dtype=dtypes, compression='gzip')

    return rideDataDf_trainX, rideDataDf_testX, rideDataDf_trainY, rideDataDf_testY


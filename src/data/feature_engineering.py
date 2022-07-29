import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import json

categoricalCols = ["WDW_TICKET_SEASON", "SEASON", "HOLIDAYN",
                   "WDWRaceN", "WDWeventN", "WDWSEASON",
                   "MKeventN", "EPeventN", "HSeventN", "AKeventN",
                   "HOLIDAYJ", "Ride_name", "Park_area",
                   "MKPRDDN", "MKPRDNN", "MKFIREN",
                   "EPFIREN", "HSPRDDN", "HSFIREN",
                   "HSSHWNN", "AKPRDDN", "AKFIREN", "AKSHWNN",
                   "Wind Quality Code", "Wind Type Code", "Wind Speed Quality",
                   "Cloud Quality Code", "Cloud Determination Code", "CAVOK Code",
                   "Visibiliy Quality Code", "Visibility Variability Code",
                   "Visibility Quality Variability Code", "Temperature Quality Code"]
bool_dtypes = [
    "Ride_type_thrill", "Ride_type_spinning", "Ride_type_slow",
    "Ride_type_small_drops", "Ride_type_big_drops", "Ride_type_dark",
    "Ride_type_scary", "Ride_type_water", "Fast_pass",
    "Classic", "Age_interest_preschoolers", "Age_interest_kids",
    "Age_interest_tweens", "Age_interest_teens", "Age_interest_adults",
    "HOLIDAY", "WDWevent", "WDWrace", "MKevent", "EPevent", "HSevent",
    "AKevent", "MKEMHMORN", "MKEMHMYEST", "MKEMHMTOM", "MKEMHEVE",
    "MKEMHEYEST", "MKEMHETOM", "EPEMHMORN", "EPEMHMYEST", "EPEMHMTOM",
    "EPEMHEVE", "EPEMHEYEST", "EPEMHETOM", "HSEMHMORN", "HSEMHMYEST",
    "HSEMHMTOM", "HSEMHEVE", "HSEMHEYEST", "HSEMHETOM", "AKEMHMORN",
    "AKEMHMYEST", "AKEMHMTOM", "AKEMHEVE", "AKEMHEYEST", "AKEMHETOM"
]
parse_dates = ['date', 'datetime']
parse_times = ["MKOPEN", "MKCLOSE", "MKEMHOPEN", "MKEMHCLOSE",
               "MKOPENYEST", "MKCLOSEYEST", "MKOPENTOM",
               "MKCLOSETOM", "EPOPEN", "EPCLOSE", "EPEMHOPEN",
               "EPEMHCLOSE", "EPOPENYEST", "EPCLOSEYEST",
               "EPOPENTOM", "EPCLOSETOM", "HSOPEN", "HSCLOSE",
               "HSEMHOPEN", "HSEMHCLOSE", "HSOPENYEST", "HSCLOSEYEST",
               "HSOPENTOM", "HSCLOSETOM", "AKOPEN", "AKCLOSE",
               "AKEMHOPEN", "AKOPENYEST", "AKCLOSEYEST",
               "AKOPENTOM", "AKCLOSETOM", "MKPRDDT1", "MKPRDDT2",
               "MKPRDNT1", "MKPRDNT2", "MKFIRET1", "MKFIRET2",
               "EPFIRET1", "EPFIRET2", "HSPRDDT1", "HSFIRET1",
               "HSFIRET2", "HSSHWNT1", "HSSHWNT2", "AKPRDDT1",
               "AKPRDDT2", "AKSHWNT1", "AKSHWNT2"]

park_metadata_cols = ["WEEKOFYEAR", "SEASON", "HOLIDAYPX", "HOLIDAYM", "HOLIDAYN", "HOLIDAY",
                      "WDWRaceN", "WDWeventN", "WDWevent",
                      "WDWrace", "WDWSEASON", "WDWMAXTEMP",
                      "WDWMINTEMP", "WDWMEANTEMP", "MKeventN",
                      "MKevent", "EPeventN", "EPevent",
                      "HSeventN", "HSevent", "AKeventN", "AKevent",
                      "HOLIDAYJ", "inSession", "inSession_Enrollment", "inSession_wdw"]


def cleanStringData(df, cols):
    """
        Cleans categorical columns in preparation for one-hot encoding
            - Fill NA with "none" - for these categorical columns an empty row usually means there is no
            event/parade/ticket season etc for that given date
            - Convert to lowercase and strip leading/trailing whitespace to deal with any inconsistencies

        Parameters
        ----------
        df : dataframe
            dataframe with all the ride data

        cols : list
            list of categorical columns to clean

        Returns
        -------
        df
            Updated dataframe
    """
    for col in cols:
        try:
            df[col] = df[col].fillna("none").apply(lambda x: x.lower().strip())
        except KeyError as e:
            print(e)
    return df


def oneHotEncoding(df_train, df_test, cols):
    """
        Takes cleaned categorical columns and applies one-hot encoding to them

        Parameters
        ----------
        df_train : dataframe
            dataframe with all the training ride data

        df_test : dataframe
            dataframe with all the testing ride data

        cols : list
            list of categorical columns to encode

        Returns
        -------
        df
            Updated dataframe
    """
    for col in reversed(cols):
        try:
            df_train[col] = df_train[col].astype("category")
            df_test[col] = df_test[col].astype("category")

        except KeyError:
            cols.remove(col)

    # Create an instance of One-hot-encoder
    enc = OneHotEncoder()
    enc_data_train = pd.DataFrame(enc.fit_transform(df_train[cols]).toarray(), dtype=bool)
    enc_data_train.columns = enc.get_feature_names_out()

    enc_data_test = pd.DataFrame(enc.transform(df_test[cols]).toarray(), dtype=bool)
    enc_data_test.columns = enc.get_feature_names_out()

    # Merge with main
    New_df_train = df_train.join(enc_data_train)
    New_df_train = New_df_train.drop(columns=cols)

    New_df_test = df_test.join(enc_data_test)
    New_df_test = New_df_test.drop(columns=cols)

    return New_df_train, New_df_test


def setVarianceThreshold(X_train, X_test, threshold, data_type):
    X = pd.concat([X_train, X_test], axis=0)
    print("X SHAPE: ", X.shape)

    X_dtype = X.select_dtypes(include=[data_type]).reset_index(drop=True)
    print("X NUM SHAPE: ", X_dtype.shape)

    var_thr = VarianceThreshold(threshold=threshold)  # Removing both constant and quasi-constant
    var_thr.fit(X_dtype)

    concol = [column for column in X_dtype.columns
              if column not in X_dtype.columns[var_thr.get_support()]]

    del var_thr, X_dtype

    print("DROPPING BOOL: ", concol)

    X_train.drop(concol, axis=1, inplace=True)
    X_test.drop(concol, axis=1, inplace=True)

    return X_train, X_test


def convertHours(x):
    """
            Takes closing times like 25:00 and converts to military time (25:00->01:00)

            Parameters
            ----------
            x : string
                string with time object (HH:MM)

            Returns
            -------
            x
                Updated time as string
        """
    try:
        if int(x[:2]) >= 24:
            if int(x[:2]) - 24 < 10:
                x = x.replace(x[:2], str(int(x[:2]) - 24))
                x = "0" + x
            else:
                x = x.replace(x[:2], str(int(x[:2]) - 24))
        return x
    except TypeError:
        return x


def trainTestSplit(year):
    """
            Reads in 1 year of clean data, splits into train/test for actual/posted wait times, respectively

            Parameters
            ----------
            year : int
                year to parse

            Returns
            -------
            2 lists of 4 dataframes each
               Returns X/y train/test for actual and posted wait times respectively
    """

    rideData = pd.read_csv(f'data/interim/RideData{year}Weather.csv', compression='gzip', dtype=dtypes,
                           parse_dates=parse_dates)
    rideData = rideData.dropna(subset=park_metadata_cols, how='all', axis=0)

    rideDataActual = rideData[~np.isnan(rideData["SACTMIN"])]
    rideDataPosted = rideData[~np.isnan(rideData["SPOSTMIN"])]

    X_actual = rideDataActual.drop(columns=["SPOSTMIN", "SACTMIN"])
    y_actual = rideDataActual["SACTMIN"]

    X_train_actual, X_test_actual, y_train_actual, y_test_actual = train_test_split(X_actual, y_actual,
                                                                                    test_size=0.33, random_state=42)

    X_posted = rideDataPosted.drop(columns=["SPOSTMIN", "SACTMIN"])
    y_posted = rideDataPosted["SPOSTMIN"]

    X_train_posted, X_test_posted, y_train_posted, y_test_posted = train_test_split(X_posted, y_posted,
                                                                                    test_size=0.33, random_state=42)

    return [X_train_actual, X_test_actual, y_train_actual, y_test_actual], \
           [X_train_posted, X_test_posted, y_train_posted, y_test_posted]


def encodeTrainAndTest(posted=True):
    allYearsTrain_X, allYearsTrain_y, allYearsTest_X, allYearsTest_y = [], [], [], []

    for year in range(2015, 2022):
        print(f"YEAR {year}")
        if posted:
            data = trainTestSplit(year)[1]
        else:
            data = trainTestSplit(year)[0]

        allYearsTrain_X.append(data[0])
        allYearsTest_X.append(data[1])
        allYearsTrain_y.append(data[2])
        allYearsTest_y.append(data[3])

    X_train = pd.concat(allYearsTrain_X, ignore_index=True)
    X_test = pd.concat(allYearsTest_X, ignore_index=True)
    y_train = pd.concat(allYearsTrain_y, ignore_index=True)
    y_test = pd.concat(allYearsTest_y, ignore_index=True)

    del allYearsTrain_X, allYearsTest_X, allYearsTrain_y, allYearsTest_y

    cleanX = []
    for df in [X_train, X_test]:
        if df is X_train:
            print("TRAIN")
        else:
            print("TEST")

        # df[bool_dtypes].astype("bool", copy=False)

        for time_col in parse_times:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format='%H:%M').dt.time

            except ValueError:
                df[time_col] = df[time_col].apply(lambda x: convertHours(x))
                df[time_col] = pd.to_datetime(df[time_col], format='%H:%M').dt.time

            except KeyError:
                continue

        dfClean = cleanStringData(df, categoricalCols)
        cleanX.append(dfClean)

    X_train, X_test = oneHotEncoding(cleanX[0], cleanX[1], categoricalCols)

    del dfClean, cleanX

    X_train, X_test = setVarianceThreshold(X_train, X_test, 0.05, np.number)
    X_train, X_test = setVarianceThreshold(X_train, X_test, 0, "bool")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    with open("data/interim/dtypes.json") as json_file:
        dtypes = json.load(json_file)

    X_train_posted, X_test_posted, y_train_posted, y_test_posted = encodeTrainAndTest(posted=True)
    X_train_actual, X_test_actual, y_train_actual, y_test_actual = encodeTrainAndTest(posted=False)

    for year in range(2015, 2022):
        print(f"WRITING {year}")
        # combing x and y together so that we keep the features and targets together
        # when splitting into smaller files

        X_train_posted["POSTED_WAIT"] = y_train_posted
        X_train_posted[X_train_posted["date"].dt.year == year].to_csv(f"data/processed/All_train_postedtimes{year}.csv",
                                                                      index=False, compression='gzip')
        X_test_posted["POSTED_WAIT"] = y_test_posted
        X_test_posted[X_test_posted["date"].dt.year == year].to_csv(f"data/processed/All_test_postedtimes{year}.csv",
                                                                    index=False, compression='gzip')

    X_train_actual.to_csv("data/processed/Xtrain_actualtimes.csv", compression='gzip')
    X_test_actual.to_csv("data/processed/Xtest_actualtimes.csv", compression='gzip')
    y_train_actual.to_csv("data/processed/ytrain_actualtimes.csv", compression='gzip')
    y_test_actual.to_csv("data/processed/ytest_actualtimes.csv", compression='gzip')

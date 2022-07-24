import numpy as np
import pandas as pd
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
    "MKEMHEYEST","MKEMHETOM", "EPEMHMORN", "EPEMHMYEST","EPEMHMTOM",
    "EPEMHEVE", "EPEMHEYEST", "EPEMHETOM",   "HSEMHMORN", "HSEMHMYEST",
    "HSEMHMTOM", "HSEMHEVE", "HSEMHEYEST", "HSEMHETOM",  "AKEMHMORN",
    "AKEMHMYEST", "AKEMHMTOM", "AKEMHEVE", "AKEMHEYEST", "AKEMHETOM"
]
parse_dates = ['date', 'datetime']
parse_times = ["MKOPEN", "MKCLOSE", "MKEMHOPEN", "MKEMHCLOSE",
               "MKOPENYEST", "MKCLOSEYEST", "MKOPENTOM",
               "MKCLOSETOM","EPOPEN", "EPCLOSE", "EPEMHOPEN",
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


park_metadata_cols = ["WEEKOFYEAR","SEASON", "HOLIDAYPX", "HOLIDAYM", "HOLIDAYN", "HOLIDAY",
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


def oneHotEncoding(df, cols):
    """
        Takes cleaned categorical columns and applies one-hot encoding to them

        Parameters
        ----------
        df : dataframe
            dataframe with all the ride data

        cols : list
            list of categorical columns to encode

        Returns
        -------
        df
            Updated dataframe
    """
    for col in reversed(cols):
        try:
            df[col] = df[col].astype("category")
        except KeyError as e:
            cols.remove(col)

    # Create an instance of One-hot-encoder
    enc = OneHotEncoder()
    enc_data = pd.DataFrame(enc.fit_transform(df[cols]).toarray(), dtype=bool)
    enc_data.columns = enc.get_feature_names_out()

    # Merge with main
    New_df = df.join(enc_data)
    New_df = New_df.drop(columns=cols)

    return New_df


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
            if int(x[:2])-24 < 10:
                x = x.replace(x[:2], str(int(x[:2])-24))
                x = "0"+x
            else:
                x = x.replace(x[:2], str(int(x[:2])-24))
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

    rideData = pd.read_csv(f'data/interim/RideData{year}Weather.csv', compression='gzip', dtype=dtypes, parse_dates=parse_dates)
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

    encodedX = []
    for df in [X_train, X_test]:
        df[bool_dtypes] = df[bool_dtypes].astype("bool")

        for time_col in parse_times:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format='%H:%M').dt.time

            except ValueError:
                df[time_col] = df[time_col].apply(lambda x: convertHours(x))
                df[time_col] = pd.to_datetime(df[time_col], format='%H:%M').dt.time

            except KeyError:
                continue

        dfClean = cleanStringData(df, categoricalCols)

        dfEncoded = oneHotEncoding(dfClean, categoricalCols)

        encodedX.append(dfEncoded)


    return encodedX[0], encodedX[1], y_train, y_test


def writeToChunkedCsvs(df, chunks, train=True, posted=True):
    for grp, each_csv in df.groupby(df.index // 4):
        each_csv.to_csv(f"test_{grp}.csv", index=False)

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

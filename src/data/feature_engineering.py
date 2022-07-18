import math
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json

categoricalCols = ["WDW_TICKET_SEASON", "SEASON", "HOLIDAYN",
                           "WDWRaceN", "WDWeventN", "WDWSEASON",
                           "MKeventN", "EPeventN", "HSeventN", "AKeventN",
                           "HOLIDAYJ", "Ride_name", "Park_area",
                           "MKPRDDN", "MKPRDNN", "MKFIREN",
                           "EPFIREN", "HSPRDDN", "HSFIREN",
                           "HSSHWNN", "AKPRDDN", "AKFIREN", "AKSHWNN"]
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
parse_dates = ['date', 'datetime', "Open_date"]
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

with open("src/data/dtypes.json") as json_file:
    dtypes = json.load(json_file)

def removeUnnecessaryColumns(df, cols):
    for col in df.columns:
        try:
            if (len(df[col].unique()) == 1) and (math.isnan(df[col].unique()[0])) and (col not in cols):
                print(f"ADDING {col} to DROP LIST")
                cols.append(col)
        except TypeError:
            continue

    df = df.drop(columns=cols)
    return df


def cleanStringData(df, cols):
    for col in cols:
        try:
            df[col] = df[col].fillna("none").apply(lambda x: x.lower().strip())
        except KeyError as e:
            print(e)
    return df


def oneHotEncoding(df, cols):
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
    if int(x[:2]) >= 24:
        if int(x[:2])-24 < 10:
            x = x.replace(x[:2], str(int(x[:2])-24))
            x = "0"+x
        else:
            x = x.replace(x[:2], str(int(x[:2])-24))
    return x


def encodeTrainAndTest(yearStart, yearEnd):
    yearsEncoded = []
    for year in range(yearStart, yearEnd+1):
        print(f"YEAR {year}")
        rideData = pd.read_csv(f'data/interim/rideData{year}.csv', compression='gzip', dtype=dtypes, parse_dates=parse_dates)
        rideData[bool_dtypes] = rideData[bool_dtypes].astype("boolean")

        for time_col in parse_times:
            try:
                rideData[time_col] = pd.to_datetime(rideData[time_col], format='%H:%M').dt.time

            except ValueError:
                rideData[time_col] = rideData[time_col].apply(lambda x: convertHours(x))
                rideData[time_col] = pd.to_datetime(rideData[time_col], format='%H:%M').dt.time

            except KeyError:
                continue

        rideData = cleanStringData(rideData, categoricalCols)

        rideDataEncoded = oneHotEncoding(rideData, categoricalCols)

        yearsEncoded.append(rideDataEncoded)

    df = pd.concat(yearsEncoded, ignore_index=True)

    return df


if __name__ == '__main__':
    trainTestSplit = [int(x) for x in sys.argv[1:5]]
    trainDf = encodeTrainAndTest(trainTestSplit[0], trainTestSplit[1])
    trainDf_POST = trainDf[~np.isnan(trainDf["SPOSTMIN"])]
    trainDf_ACT = trainDf[~np.isnan(trainDf["SACTMIN"])]

    trainDf_POST.to_csv("data/processed/trainPostedTimes.csv", compression='gzip')
    trainDf_ACT.to_csv("data/processed/trainActualTimes.csv", compression='gzip')

    testDf = encodeTrainAndTest(trainTestSplit[2], trainTestSplit[3])
    testDf_POST = testDf[~np.isnan(testDf["SPOSTMIN"])]
    testDf_ACT = testDf[~np.isnan(testDf["SACTMIN"])]

    print(dict(testDf_POST.iloc[0]))

    testDf_POST.to_csv("data/processed/testPostedTimes.csv", compression='gzip')
    testDf_ACT.to_csv("data/processed/testActualTimes.csv", compression='gzip')

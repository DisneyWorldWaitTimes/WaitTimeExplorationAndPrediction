import json
import pandas as pd
import numpy as np
from helper import *
from weather_data import weatherData
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def stringPercentToInt(df):
    """
    Convert string percentage columns (inSession_*) to int8.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to parse (in our case the park_metadata dataframe)

    Returns
    -------
    df
        Updated dataframe

    """
    for col in df.columns:
        if str.startswith(col.lower(), "insession"):
            df[col] = df[col].apply(lambda x: x.strip("%") if type(x) == str else x)
            df[col] = df[col].str.strip().astype(float).astype('Int8')

    return df


def yesNoToBool(df):
    """
    Convert string columns with "Yes" or "No" values to boolean.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to parse (in our case the DataWorld dataframe)

    Returns
    -------
    None
        Updates the dataframe in place.

    """

    for col in df.columns:
        col_values = sorted([x.lower() for x in df[col].unique() if type(x) == str])
        if (col_values == ["no", "yes"]) | (col_values == ["no"]) | (col_values == ["yes"]):
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    del col_values


def combineRidesAndName(ride_files, ride_names, data_world_df):
    """
    Combine ride files with data.world data

    Parameters
    ----------
    ride_files : list
        list of csv files with wait times to parse

    ride_names : list
        list of Ride names - must match order of ride_files & data.world ride name

    data_world_df : DataFrame
        dataframe of ride metadata from data.world

    Returns
    -------
    DataFrame
        Returns dataframe with all rides merged with their respective data.world metadata

    """

    all_rides_with_dw_metadata = []

    for idx, ride in enumerate(ride_files):
        ride_waits = pd.read_csv(ride)
        ride_waits["Ride_name"] = ride_names[idx]

        ride_waits_with_metadata = ride_waits.merge(data_world_df, how="left")
        all_rides_with_dw_metadata.append(ride_waits_with_metadata)

    del ride_waits_with_metadata

    return pd.concat(all_rides_with_dw_metadata, ignore_index=True)


def dateCleaning(df):
    """
    Clean & update date columns:
        datetime - converted to DateTime datatype
        date - converted to DateTime datatype
        Age_of_ride_days - updated to be the age of ride in days on that exact date in the park
        Age_of_ride_years - updated to be the age of ride in years on that exact date in the park (Age_of_ride_days/365)
        Age_of_ride_total  - dropped because not useful for model


    Parameters
    ----------
    df : DataFrame
        dataframe with data.world columns and ride times

    Returns
    -------
    None
        updates df in place

    """

    df['datetime'] = pd.to_datetime(df["datetime"])
    df['date'] = pd.to_datetime(df["date"])

    df["Age_of_ride_days"] = (df["date"] - df["Open_date"]).dt.days.astype('int16')
    df["Age_of_ride_years"] = df["Age_of_ride_days"] / 365

    df.drop("Age_of_ride_total", axis=1, inplace=True)

    return df


def combineCovidData(rideData):
    """
        Extract & combine covid-data (number of daily US new cases) into existing park metadata by day

        Parameters
        ----------
        rideData : DataFrame
            dataframe with wait times & metadata

        Returns
        -------
        rides_with_covid: Dataframe
            updated dataframe with appended covid metrics

    """
    covidData = pd.read_csv("data/raw/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")

    covidData = covidData.groupby("DATE")["new_case"].sum().reset_index()
    covidData["DATE"] = pd.to_datetime(covidData["DATE"])
    rides_with_covid = rideData.merge(covidData, on="DATE", how='left')
    rides_with_covid["new_case"] = rides_with_covid["new_case"].fillna(0)

    del covidData

    return rides_with_covid


def combineMetadataAndUpdate(ride_files, ride_names):
    """
        Combine all metadata together with respective dates & rides

        Parameters
        ----------
        ride_files : list
            list of csv files with wait times to parse (input to cleanData())

        ride_names : list
            list of Ride names - must match order of ride_files & data.world ride name (input to cleanData())

        Returns
        -------
        combined_data : DataFrame
            Returns updated dataframe with all rides merged with their respective daily park & ride metadata

    """
    park_metadata = pd.read_csv("data/raw/park_metadata.csv")
    data_world = pd.read_excel("data/raw/WDW_Ride_Data_DW.xlsx")
    mk_dw = data_world[data_world["Park_location"] == "MK"]
    del data_world

    park_metadata = stringPercentToInt(park_metadata)
    yesNoToBool(mk_dw)  # convert Yes/No columns to boolean

    all_rides = combineRidesAndName(ride_files,
                                    ride_names, mk_dw)  # combine wait time data with data.world metadata

    del mk_dw
    dateCleaning(all_rides)  # clean date columns


    all_rides['DATE'] = all_rides['date']
    park_metadata['DATE'] = pd.to_datetime(park_metadata["DATE"])
    print("ALL RIDES SHAPE: ", all_rides.shape)

    combined_data = all_rides.merge(park_metadata, how="left", on="DATE")
    combined_data = combineCovidData(combined_data)
    combined_data = combined_data.drop(columns=['DATE', "WDWTICKETSEASON", "Park_location",
                                                "Ride_type_all", "Age_interest_all"])

    print("COMBINED DATA SHAPE: ", combined_data.shape)

    del all_rides, park_metadata

    return combined_data


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

        except KeyError as e:
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
    del df_train, enc_data_train

    New_df_test = df_test.join(enc_data_test)
    New_df_test = New_df_test.drop(columns=cols)
    del df_test, enc_data_test

    return New_df_train, New_df_test

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


def setVarianceThreshold(X_train, X_test, threshold, data_type):
    """
        Removing columns below variance threshold limit for a given datatype

        Parameters
        ----------
        X_train : dataframe
            Training features for model
        X_test : dataframe
            Testing features for model
        threshold: Integer
            Threshold to set for variance
        data_type : string
            Specifies which datatype columns to pass into the variance threshold object

        Returns
        -------
        X_train, X_test
            Resulting train/test feature sets after variance threshold tuning
    """

    # X = pd.concat([X_train, X_test], axis=0)

    X_dtype = X_train.select_dtypes(include=[data_type]).reset_index(drop=True)

    var_thr = VarianceThreshold(threshold=threshold)  # Removing both constant and quasi-constant
    var_thr.fit(X_train)

    concol = [column for column in X_dtype.columns
              if column not in X_dtype.columns[var_thr.get_support()]]

    del var_thr, X_dtype

    if "Weather Type" in concol:
        concol.remove("Weather Type")

    print(f"DROPPING {data_type}: ", concol)
    X_train.drop(concol, axis=1, inplace=True)
    X_test.drop(concol, axis=1, inplace=True)

    return X_train, X_test


def trainTestSplit(input_dir, year):
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

    rideData = pd.read_csv(f'{input_dir}/RideData{year}Weather.csv', compression='gzip', dtype=dtypes, parse_dates=parse_dates)
    rideData = rideData.dropna(subset=park_metadata_cols, how='all', axis=0)

    rideDataActual = rideData[~np.isnan(rideData["SACTMIN"])]
    rideDataPosted = rideData[~np.isnan(rideData["SPOSTMIN"])]

    del rideData

    X_actual = rideDataActual.drop(columns=["SPOSTMIN", "SACTMIN"])
    y_actual = rideDataActual["SACTMIN"]

    X_train_actual, X_test_actual, y_train_actual, y_test_actual = train_test_split(X_actual, y_actual,
                                                                 test_size=0.33, random_state=42)

    X_posted = rideDataPosted.drop(columns=["SPOSTMIN", "SACTMIN"])
    y_posted = rideDataPosted["SPOSTMIN"]

    X_train_posted, X_test_posted, y_train_posted, y_test_posted = train_test_split(X_posted, y_posted,
                                                                                    test_size=0.33, random_state=42)

    del rideDataActual, rideDataPosted, X_actual, X_posted, y_actual, y_posted

    return [X_train_actual, X_test_actual, y_train_actual, y_test_actual], \
               [X_train_posted, X_test_posted, y_train_posted, y_test_posted]


def encodeTrainAndTest(input_dir, posted=True):
    """
        Put it all together to clean train and test datasets

        Parameters
        ----------
        input_dir : String
            Directory where input data is located

        Returns
        -------
        X_train, X_test, y_train, y_test: DataFrames
           Cleaned, encoded, & combined dataframes for train/test features & targets
    """
    allYearsTrain_X, allYearsTrain_y, allYearsTest_X, allYearsTest_y = [], [], [], []

    for year in range(2015, 2022):
        print(f"YEAR {year}")
        if posted:
            data = trainTestSplit(input_dir, year)[1]
        else:
            data = trainTestSplit(input_dir, year)[0]

        allYearsTrain_X.append(data[0])
        allYearsTest_X.append(data[1])
        allYearsTrain_y.append(data[2])
        allYearsTest_y.append(data[3])

    X_train = pd.concat(allYearsTrain_X, ignore_index=True)
    X_test = pd.concat(allYearsTest_X, ignore_index=True)
    y_train = pd.concat(allYearsTrain_y, ignore_index=True)
    y_test = pd.concat(allYearsTest_y, ignore_index=True)

    del allYearsTrain_X, allYearsTest_X, allYearsTrain_y, allYearsTest_y

    # parse time columns to datetime objects
    cleanX = []
    for df in [X_train, X_test]:
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

    # one hot encoded the categorical columns
    X_train, X_test = oneHotEncoding(cleanX[0], cleanX[1], categoricalCols)

    del dfClean, cleanX

    # set the variance threshold for training and testing data for boolean and numeric columsn
    X_train, X_test = setVarianceThreshold(X_train, X_test, 0.05, np.number)
    X_train, X_test = setVarianceThreshold(X_train, X_test, 0.001, "bool")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import argparse

    # parse input and output args
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Interim Data Directory")
    parser.add_argument('output', help="Processed Data Directory")
    args = parser.parse_args()

    # open dtypes to specify when loading CSVs
    with open(f"{args.input}/dtypes.json") as json_file:
        dtypes = json.load(json_file)

    combined_data = combineMetadataAndUpdate(ride_files, ride_names)
    for year in range(2015, 2022):
        weatherData(combined_data, year)

    del combined_data, year

    for postedActual in ["posted", "actual"]:
        if postedActual == "posted":
            print("posted")
            X_train, X_test, y_train, y_test = encodeTrainAndTest(args.input, posted=True)
            # combing x and y together so that we keep the features and targets together
            # when splitting into smaller files
            X_train["POSTED_WAIT"] = y_train
            X_test["POSTED_WAIT"] = y_test

            for year in range(2015, 2022):
                print(f"WRITING {year}")
                # write to year based files so that the data will fit on GitHub
                X_train[X_train["date"].dt.year == year].to_csv(
                    f"{args.output}/All_train_postedtimes{year}.csv",
                    index=False, compression='gzip')

                X_test[X_test["date"].dt.year == year].to_csv(
                    f"{args.output}/All_test_postedtimes{year}.csv",
                    index=False, compression='gzip')

            del X_train, y_train, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = encodeTrainAndTest(posted=False)

            X_train.to_csv(f"{args.output}/Xtrain_actualtimes.csv", compression='gzip')
            del X_train

            X_test.to_csv(f"{args.output}/Xtest_actualtimes.csv", compression='gzip')
            del X_test

            y_train.to_csv(f"{args.output}/ytrain_actualtimes.csv", compression='gzip')
            del y_train

            y_test.to_csv(f"{args.output}/ytest_actualtimes.csv", compression='gzip')
            del y_test


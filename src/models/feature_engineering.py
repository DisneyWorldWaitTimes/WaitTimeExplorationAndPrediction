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
    """
        Loads train test data for posted wait times

        Parameters
        ----------
        input_dir: string
            Directory where the cleaned posted wait time datasets are located (data/processed)

        Returns
        -------
        ride_data_df_train_x - train data features for posted wait times
        ride_data_df_test_x - test data features for posted wait times
        ride_data_df_train_y - train data targets for posted wait times
        ride_data_df_test_y - test data targets for posted wait times

    """

    print("LOADING DATA")
    parse_dates = ['date', 'datetime']
    X_train_list = []
    y_train_list = []
    dtypes = setup()

    for year in range(2015, 2022):
        ride_data = pd.read_csv(f'{input_dir}/All_train_postedtimes{year}.csv', dtype=dtypes,
                               parse_dates=parse_dates, compression='gzip')
        ride_data_x = ride_data.drop(columns=["POSTED_WAIT"])
        ride_data_y = ride_data["POSTED_WAIT"]
        X_train_list.append(ride_data_x)
        y_train_list.append(ride_data_y)

    ride_data_df_train_x = pd.concat(X_train_list, ignore_index=True)
    ride_data_df_train_y = pd.concat(y_train_list, ignore_index=True)

    X_test_list = []
    y_test_list = []

    for year in range(2015, 2022):
        ride_data = pd.read_csv(f'{input_dir}/All_test_postedtimes{year}.csv', dtype=dtypes,
                               parse_dates=parse_dates, compression='gzip')
        ride_data_x = ride_data.drop(columns=["POSTED_WAIT"])
        ride_data_y = ride_data["POSTED_WAIT"]

        X_test_list.append(ride_data_x)
        y_test_list.append(ride_data_y)

    ride_data_df_test_x = pd.concat(X_test_list, ignore_index=True)
    ride_data_df_test_y = pd.concat(y_test_list, ignore_index=True)

    return ride_data_df_train_x, ride_data_df_test_x, ride_data_df_train_y, ride_data_df_test_y


def data_preparation_for_pipeline(X_train, X_test, y_train, y_test):
    """
            Converts datetime objects to integer representations:
                MONTHOFYEAR, DAYOFYEAR, YEAR, & HOUROFDAY

            Sorts in date order so that next step of data imputation is backfilling based on similar entries

            Parameters
            ----------
            X_train, X_test, y_train, y_test: DataFrames
                Output dataframes of load_train_test_posted_wait_times

            Returns
            -------
            X_train_clean, X_test_clean, y_train, y_test: DataFrames

        """

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


if __name__ == '__main__':
    import argparse

    # parse input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Clean Train/Test Data")
    parser.add_argument('output', help="Engineered data directory")
    args = parser.parse_args()

    # load in data - results of data cleaning step & final feature engineering before pipeline
    X_train, X_test, y_train, y_test = load_train_test_posted_wait_times(args.input)
    X_train_clean, X_test_clean, y_train, y_test = data_preparation_for_pipeline(X_train, X_test, y_train, y_test)

    # write files to data/final for pipeline
    print("WRITING FILES")
    for idx, file in enumerate([X_train_clean, X_test_clean, y_train, y_test]):
        names = ["X_train", "X_test", "y_train", "y_test"]
        file.to_csv(f"{args.output}/{names[idx]}_posted_final.csv", compression='gzip')

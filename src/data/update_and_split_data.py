import pandas as pd
import json

ride_files = ['data/raw/7_dwarfs_train.csv', 'data/raw/astro_orbiter.csv', 'data/raw/barnstormer.csv',
              'data/raw/big_thunder_mtn.csv', 'data/raw/buzz_lightyear.csv',
              'data/raw/carousel_of_progress.csv', 'data/raw/dumbo.csv',
              'data/raw/haunted_mansion.csv', 'data/raw/it_s_a_small_world.csv',
              'data/raw/jungle_cruise.csv', 'data/raw/mad_tea_party.csv', 'data/raw/magic_carpets.csv',
              'data/raw/main_st_vehicles.csv', 'data/raw/peoplemover.csv',
              'data/raw/peter_pan_s_flight.csv',
              'data/raw/pirates_of_caribbean.csv', 'data/raw/regal_carrousel.csv',
              'data/raw/space_mountain.csv',
              'data/raw/splash_mountain.csv', 'data/raw/tom_land_speedway.csv',
              'data/raw/winnie_the_pooh.csv']

ride_names = ['Seven Dwarfs Mine Train', 'Astro Orbiter', 'The Barnstormer', 'Big Thunder Mountain Railroad',
              "Buzz Lightyear's Space Ranger Spin", "Walt Disney's Carousel of Progress",
              'Dumbo the Flying Elephant',
              'Haunted Mansion', "It's a Small World", 'Jungle Cruise', 'Mad Tea Party',
              'The Magic Carpets of Aladdin',
              'Main Street Vehicles', 'Tomorrowland Transit Authority PeopleMover', "Peter Pan's Flight",
              'Pirates of the Caribbean', 'Prince Charming Regal Carrousel', 'Space Mountain', 'Splash Mountain',
              'Tomorrowland Speedway', 'The Many Adventures of Winnie the Pooh']

with open("src/data/dtypes.json") as json_file:
    dtypes = json.load(json_file)

def convertToInt8(df):
    """
    Converts int columns with finite values to int8

    Parameters
    ----------
    df : DataFrame
        The DataFrame to parse (in our case the park_metadata dataframe)

    Returns
    -------
    df
        Updated df with compressed dtypes

    """
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'int32':
            if sorted(df[col].unique()) == [0, 1]:
                df[col] = df[col].astype('int8')

    for x in ['DAYOFWEEK', 'DAYOFYEAR', 'WEEKOFYEAR', 'MONTHOFYEAR']:
        df[x] = df[x].astype('int8')

    return df


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


def combineCovidData(rideData):
    covidData = pd.read_csv("data/raw/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")

    covidData = covidData.groupby("DATE")["new_case"].sum().reset_index()
    covidData["DATE"] = pd.to_datetime(covidData["DATE"])
    rides_with_covid = rideData.merge(covidData, on="DATE", how='left')
    rides_with_covid["new_case"] = rides_with_covid["new_case"].fillna(0)

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

    park_metadata = stringPercentToInt(park_metadata)
    park_metadata = convertToInt8(park_metadata)
    yesNoToBool(mk_dw)  # convert Yes/No columns to boolean

    all_rides = combineRidesAndName(ride_files,
                                    ride_names, mk_dw)  # combine wait time data with data.world metadata

    dateCleaning(all_rides)  # clean date columns


    all_rides['DATE'] = all_rides['date']
    park_metadata['DATE'] = pd.to_datetime(park_metadata["DATE"])
    print("ALL RIDES SHAPE: ", all_rides.shape)

    combined_data = all_rides.merge(park_metadata, how="left", on="DATE")
    combined_data = combineCovidData(combined_data)
    combined_data = combined_data.drop(columns=['DATE', "WDWTICKETSEASON", "Park_location",
                                                "Ride_type_all", "Age_interest_all"])

    print("COMBINED DATA SHAPE: ", combined_data.shape)

    return combined_data


def seperateYearsAndWriteToCSV(df, year):
    """
        Separates full dataset into 2 month increments and writes to CSV files in data/interim

        Parameters
        ----------
        df : dataframe
            dataframe with all data for all 21 rides

        year : int
            year of data to parse

        Returns
        -------
        None
            Writes data to data/interim

    """

    df_year = df[df['date'].dt.year == year]
    df_year.to_csv(f'data/interim/rideData{year}.csv', compression='gzip')


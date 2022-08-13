from weather_helpers import *
import glob, os
import pandas as pd


def weatherData(Ride_data, year):
    Ride_data = Ride_data[Ride_data["datetime"].dt.year == year]

    # for file in glob.glob("* Weather.csv"):
    Weather_data = pd.read_csv(f'data/interim/{year}Weather.csv')
    Weather_data = Weather_data[['DATE', 'SOURCE', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND', 'CIG',
                                 'VIS', 'TMP', 'DEW', 'SLP', "AT1"]]

    # converting concatenated columns into separate columns
    Weather_data[['Wind Angle', 'Wind Quality Code', 'Wind Type Code', 'Wind Speed', "Wind Speed Quality"
                  ]] = Weather_data['WND'].str.split(',', 5, expand=True)
    Weather_data[['Cloud Height', 'Cloud Quality Code', 'Cloud Determination Code', 'CAVOK Code'
                  ]] = Weather_data['CIG'].str.split(',', 4, expand=True)
    Weather_data[['Visibility Distance (M)', 'Visibiliy Quality Code', 'Visibility Variability Code',
                  'Visibility Quality Variability Code']] = Weather_data['VIS'].str.split(',', 4, expand=True)
    Weather_data[['Temperature (C)', 'Temperature Quality Code']] = Weather_data['TMP'].str.split(',', 2,
                                                                                                  expand=True)
    Weather_data[['Source Element', 'Weather Type', 'Weather Type Observation',
                  'Weather Code Quality Code']] = Weather_data['AT1'].str.split(',', 4, expand=True)

    # applying dictionary replacements for values to more human-readable ones
    Weather_data = Weather_data.replace({'Wind Quality Code': WQCDict})
    Weather_data = Weather_data.replace({'Wind Speed Quality': WSQDict})
    Weather_data = Weather_data.replace({'Wind Type Code': WTCDict})
    Weather_data = Weather_data.replace({'Cloud Quality Code': CQCDict})
    Weather_data = Weather_data.replace({'Cloud Determination Code': CDCDict})
    Weather_data = Weather_data.replace({'CAVOK Code': CAVOKDict})
    Weather_data = Weather_data.replace({'Visibiliy Quality Code': VQCDict})
    Weather_data = Weather_data.replace({'Visibility Variability Code': VVCDict})
    Weather_data = Weather_data.replace({'Visibility Quality Variability Code': VQVCDict})
    Weather_data = Weather_data.replace({'Temperature Quality Code': TQCDict})
    Weather_data = Weather_data.replace({'Source Element': SEDict})
    Weather_data = Weather_data.replace({'Weather Type': WTDict})
    Weather_data = Weather_data.drop(['Weather Type Observation'], axis=1)
    Weather_data = Weather_data.replace({'Weather Code Quality Code': WCQCDict})
    Weather_data = Weather_data.fillna(0)

    # importing
    Ride_data['datetime'] = pd.to_datetime(Ride_data['datetime'])
    Ride_data['round_hour'] = Ride_data['datetime'].dt.floor('h')
    Weather_data["datetime"] = pd.to_datetime(Weather_data['DATE'])
    Weather_data['round_hour'] = Weather_data['datetime'].dt.floor('h')
    Weather_data = Weather_data.drop_duplicates(subset=['round_hour'])
    updated_file = Ride_data.merge(Weather_data, on='round_hour', how='left')
    updated_file = updated_file.rename(columns={'datetime_x': 'datetime'})
    updated_file = updated_file.drop(columns=['Open_date', 'datetime_y', 'round_hour', 'DATE', 'SOURCE',
                                              'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND',
                                              'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AT1'])
    updated_file.to_csv('data/interim/RideData' + str(year) + 'Weather.csv', compression='gzip')

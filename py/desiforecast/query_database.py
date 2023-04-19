"""This script will query DESI telemetry databases and update any local pickled data. Tables and columns of interest are included below and can be modified to get additional data as needed.
"""

import gc
import os
import psycopg2
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(
                    prog = 'query_database',
                    description = 'This program will submit SQL queries to get DESI telemetry data')
parser.add_argument('-m', '--machine-learning',
                    action='store_true',
                    help='Query the database and resample as needed for machine learning applications')
args = parser.parse_args()
data_dir = f'./data'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)


def save(data: pd.DataFrame, table: str) -> None:
    """Ramples DESI telemetry data contained within Pandas dataframe.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Queried data from the DESI telemetry database
    
    table : str
        Table name of data to save
    
    Returns
    -------
    None
    """
    data = fits.table_to_hdu(Table.from_pandas(data))
    data.writeto(f'{data_dir}/{table}.fits', overwrite=True)

def load(rows: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Preprocesses DESI telemetry data contained within Pandas dataframe.
    
    Parameters
    ----------
    rows : numpy.ndarray
        Unprocessed data from the DESI telemetry database
    
    columns : list[str]
        column names that were queired from database
    
    Returns
    -------
    data : pandas.DataFrame
        Resampled and interpolated DESI telemetry data
    """
    data = pd.DataFrame(rows, columns=columns)
    data.set_index('time_recorded', inplace=True)
    data.index = data.index.strftime('%Y-%m-%dT%H:%M:%S.%f')
    data = data.replace({pd.NA: np.nan})
    data.index.name = 'time'
    data.sort_index(inplace=True)
    data.reset_index(inplace=True)
    data['time'] = data['time'].astype(str)
    return data

def resample(rows: np.ndarray, columns: list[str]) -> pd.DataFrame :
    """Ramples DESI telemetry data contained within Pandas dataframe.
    
    Parameters
    ----------
    rows : numpy.ndarray
        Unprocessed data from the DESI telemetry database
    
    columns : list[str]
        column names that were queired from database
    
    Returns
    -------
    data : pandas.DataFrame
        Resampled and interpolated DESI telemetry data
    """
    data.set_index('time_recorded', inplace=True)
    data.index.name = 'time'
    data.interpolate(limit_direction='both', inplace=True)
    data = data.resample('6S').mean()
    data = data.tz_convert("America/Phoenix")
    return data

def main() -> None:
    """Main function to facilitate queries of DESI telemetry data
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    
    # Data table names and respective columns of interest
    labels = {}
    labels['environmentmonitor_tower'] = ['time_recorded', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    labels['environmentmonitor_dome'] = ['time_recorded', 'dome_left_upper', 'dome_left_lower', 'dome_right_upper', 'dome_right_lower', 'dome_back_upper', 'dome_back_lower', 'dome_floor_ne', 'dome_floor_nw', 'dome_floor_s']
    labels['environmentmonitor_telescope'] = ['time_recorded', 'mirror_avg_temp', 'mirror_desired_temp', 'mirror_temp', 'mirror_cooling', 'air_temp', 'air_flow', 'air_dewpoint']
    if not args.machine_learning:
        labels['etc_seeing'] = ['time_recorded', 'etc_seeing', 'seeing']
        labels['etc_telemetry'] = ['time_recorded', 'seeing', 'transparency', 'skylevel']
        labels['tcs_info'] = ['time_recorded', 'mirror_ready', 'airmass']
    
    # Establish connection to server and begin SQL queries using labels shown above.
    host = 'replicator-db.desi.lbl.gov'
    port = '5432'
    database = 'desi_dev'
    user = 'desi_reader'
    password = 'reader'
    data = pd.DataFrame()
    for table, columns in tqdm(labels.items(), desc='Querying database tables...'):
        conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        with conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT {", ".join(columns)} FROM {table};')
                rows = np.asarray(cur.fetchall())
                if args.machine_learning:
                    if data.empty:
                        data = resample(rows=rows, columns=columns)
                    else:
                        data = pd.merge(data, resample_data(rows=rows, columns=columns))
                else:
                    data = load(rows=rows, columns=columns)
                    save(data, table=table)
    if args.machine_learning:
        data.reset_index(inplace=True)
        data['time'] = data['time'].astype(str)
        save(data=data, table='desiforecast_ML_data')

if __name__ == '__main__':
    main()
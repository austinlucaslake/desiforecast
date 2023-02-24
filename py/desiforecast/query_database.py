"""This script will query DESI telemetry databases and update any local pickled data. Tables and columns of interest are included below and can be modified to get additional data as needed.
"""

import gc
import os
import psycopg2
import pandas as pd
import numba as nb
import pickle
from tqdm import tqdm
from datetime import datetime
import pytz
import argparse
parser = argparse.ArgumentParser(
                    prog = 'query_database',
                    description = 'This program will submit SQL queries to get DESI telemetry data')
parser.add_argument('-m', '--merge',
                    action='store_true',
                    help='merged and resampled data tables')
args = parser.parse_args()

def resample_data(df):
    """Resamples DESI telemetry data contained within Pandas dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Original data as queried from the DESI telemetry database
    
    Returns
    -------
    df : pandas.DataFrame
        Resampled and interpolated DESI telemetry data
    """
    if 'seeing' in df.columns.values:
        pass
    else:
        df = df.resample('6S').mean()
        df.interpolate(limit_direction='both', inplace=True)
    df = df.tz_convert("America/Phoenix")
    return df

def main():
    """Main function to facilitate queries
    
    Parameters
    ----------
        None
    
    Returns
    -------
        None
    """
   
    directory = f'./data'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    # Data table names and respective columns of interest
    labels = {}
    labels['environmentmonitor_tower'] = ['time_recorded', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'dimm', 'dewpoint']
    labels['environmentmonitor_dome'] = ['time_recorded', 'dome_left_upper', 'dome_left_lower', 'dome_right_upper', 'dome_right_lower', 'dome_back_upper', 'dome_back_lower', 'dome_floor_ne', 'dome_floor_nw', 'dome_floor_s']
    labels['environmentmonitor_telescope'] = ['time_recorded', 'mirror_avg_temp', 'mirror_desired_temp', 'mirror_temp', 'mirror_cooling', 'air_temp', 'air_flow', 'air_dewpoint']
    labels['etc_seeing'] = ['time_recorded', 'etc_seeing', 'seeing']
    labels['etc_telemetry'] = ['time_recorded', 'seeing', 'transparency', 'skylevel']
    labels['tcs_info'] = ['time_recorded', 'mirror_ready', 'airmass']
    
    # Establish connection to server and begin SQL queries using labels shown above.
    host = 'replicator-db.desi.lbl.gov'
    port = '5432'
    database = 'desi_dev'
    user = 'desi_reader'
    password = 'reader'
    first = True
    data = pd.DataFrame()
    for table, columns in tqdm(labels.items(), desc='Querying database tables...'):
        conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        with conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT {", ".join(columns)} FROM {table};')
                df = pd.DataFrame(cur.fetchall())
                df.columns = columns
                df.set_index('time_recorded', inplace=True)
                df.sort_index(inplace=True)
                if args.merge:
                    df = resample_data(df)
                    if first:
                        data = df
                    else:
                        data = data.join(df)
                else:
                    with open(f'{directory}/{table}.pkl', 'wb') as pf:
                        pickle.dump(df, pf)
                print(f'Queried {table}!')
        first = False
    if args.merge:
        with open(f'{directory}/desi_telemetry_data.pkl', 'wb') as pf:
            pickle.dump(df, pf)

if __name__ == '__main__':
    main()
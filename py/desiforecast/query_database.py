"""This script will query DESI telemetry databases and update any local pickled data. Tables and columns of interest are included below and can be modified to get additional data as needed.
"""

import psycopg2
import pandas as pd
import numba as nb
import pickle
import logging
from tqdm import tqdm
from datetime import datetime
import pytz

def preprocess_data(df):
    """Preprocesses DESI telemetry data contained within Pandas dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Unprocessed data as queried from the DESI telemetry database
    
    Returns
    -------
    df : pandas.DataFrame
        Resampled and interpolated DESI telemetry data
    """
    df.set_index('time_recorded', inplace=True)
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
   
    logging.basicConfig(filename=f'data/query.log')
    logger = logging.getLogger()
    logger.info(f'Query date: {datetime.now(pytz.timezone("America/Phoenix"))}')
    logger.info(f'Preprocessed: {args.preprocess}\n')
    
    # Data table names and respective columns of interest
    labels = {}
    labels['environmentmonitor_tower'] = ['time_recorded', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'dimm', 'dewpoint']
    labels['environmentmonitor_dome'] = ['time_recorded', 'dome_left_upper', 'dome_left_lower', 'dome_right_upper', 'dome_right_lower', 'dome_back_upper', 'dome_back_lower', 'dome_floor_ne', 'dome_floor_nw', 'dome_floor_s']
    labels['environmentmonitor_telescope'] = ['time_recorded', 'mirror_avg_temp', 'mirror_desired_temp', 'mirror_temp', 'mirror_cooling', 'air_temp', 'air_flow', 'air_dewpoint']
    labels['etc_seeing'] = ['time_recorded', 'etc_seeing', 'seeing']
    labels['etc_telemetry'] = ['time_recorded', 'seeing', 'transparency', 'skylevel']
    labels['tcs_info'] = ['time_recorded', 'mirror_ready', 'airmass']


    # Establish connection to server and begin SQL queries using labels shown above.
    try:
        host = 'replicator-db.desi.lbl.gov'
        database = 'desi_dev'
        user = 'desi_reader'
        password = 'reader'
        for table, columns in tqdm(labels.items(), desc='Querying database table...'):
            conn = psycopg2.connect(host=host, port='5432', database=database, user=user, password=password)
            logger.info(f'{table}:')
            with conn:
                with conn.cursor() as cur:
                    logger.info('\tQuerying database table...')
                    cur.execute(f'SELECT {", ".join(columns)} FROM {table};')
                    df = pd.DataFrame(cur.fetchall())
                    df.columns = columns
                    logger.info('\tPreprocessing data...')
                    df = preprocess_data(df)
                    logger.info('\tSaving to file...')
                    with open(f'data/{table}.pkl', 'wb') as pf:
                        pickle.dump(df, pf)
                    logger.info('\tQuery complete!')
    except Exception as e:
        logger.error(f'\nQuery failed!\n')
        logger.exception(f'{e}\n')


if __name__ == '__main__':
    main()
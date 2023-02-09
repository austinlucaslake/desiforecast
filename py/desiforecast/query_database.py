"""
This script will query DESI telemetry databases and update any local pickled data. Tables and columns of interest are included below and can be modified to get additional columns as needed.
"""

import psycopg2
import pandas as pd
import cudf 

def preprocess(rows, columns):
    df = 
    df.set_index('time_recorded', inplace=True)
    df = df.resample('6S').mean().tz_convert("America/Phoenix")
    df.interpolate(limit_direction='both', inplace=True)
    return cudf.from_pandas(df)

if __name__ == '__main__':
    # Data table names and respective columns of interest
    labels = {}
    labels['environmentmonitor_tower'] = ['time_recorded', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'dimm', 'dewpoint']
    labels['environmentmonitor_dome'] = ['time_recorded', 'dome_left_upper', 'dome_left_lower', 'dome_right_upper', 'dome_right_lower', 'dome_back_upper', 'dome_back_lower', 'dome_floor_ne', 'dome_floor_nw', 'dome_floor_s']
    labels['environmentmonitor_telescope'] = ['time_recorded', 'mirror_status', 'mirror_avg_temp', 'mirror_desired_temp', 'mirror_temp', 'mirror_cooling', 'air_temp', 'air_flow', 'air_dewpoint']
    labels['etc_seeing'] = ['time_recorded', 'etc_seeing', 'seeing']
    labels['etc_telemetry'] = ['time_recorded', 'seeing', 'transparency', 'skylevel']
    labels['tcs_info'] = ['time_recorded', 'mirror_ready', 'airmass']

    # Establish connection to server and begin SQL queries using labels shown above.
    conn = psycopg2.connect(host='replicator-db.desi.lbl.gov', port='5432', database='desi_dev', user='desi_reader', password='reader')
    print('# of rows for each column')
    for table, columns in labels.items():
        with conn:
            with conn.cursor() as cur:
                select = f'SELECT {", ".join(columns)} FROM {table};'
                cur.execute(select)
                df = pd.DataFrame()
                df.values = cur.fetchall()
                df.columns = columns
                df = preprocess(df)
                with open(f'data/{table}.pkl', 'wb') as pf:
                    pickle.dump(df, pf) 
    conn.close()

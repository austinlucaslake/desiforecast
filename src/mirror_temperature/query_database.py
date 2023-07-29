from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

from mirror_temperature.settings import Settings


def save(output_folder: Path, data: pd.DataFrame, table: str) -> None:
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
    data.to_pickle(output_folder / f"{table}.pkl")


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

    # Note: not entirely clear _why_ we are making some of these transformations,
    # but it is likely required to save into FITS format later.
    data = (
        pd.DataFrame(
            rows,
            columns=columns
        )
        .sort_values("time_recorded")
        .replace({pd.NA: np.nan})
        .rename({"time_recorded": "time"})
        .assign(
            time=lambda df: df.strftime("%Y-%m-%dT%H:%M:%S.%f").astype(str)
        )
    )
    return data


def retrieve_and_store_data_from_database(settings: Settings) -> None:
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
    labels["environmentmonitor_tower"] = [
        "time_recorded",
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_direction",
    ]
    labels["environmentmonitor_dome"] = [
        "time_recorded",
        "dome_left_upper",
        "dome_left_lower",
        "dome_right_upper",
        "dome_right_lower",
        "dome_back_upper",
        "dome_back_lower",
        "dome_floor_ne",
        "dome_floor_nw",
        "dome_floor_s",
    ]
    labels["environmentmonitor_telescope"] = [
        "time_recorded",
        "mirror_avg_temp",
        "mirror_desired_temp",
        "mirror_temp",
        "mirror_cooling",
        "air_temp",
        "air_flow",
        "air_dewpoint",
    ]
    labels["etc_seeing"] = ["time_recorded", "etc_seeing", "seeing"]
    labels["etc_telemetry"] = ["time_recorded", "seeing", "transparency", "skylevel"]
    labels["tcs_info"] = ["time_recorded", "mirror_ready", "airmass"]

    # Establish connection to server and begin SQL queries using labels shown above.
    data = pd.DataFrame()
    for table, columns in tqdm(labels.items(), desc="Querying database tables..."):
        conn = psycopg2.connect(
            host=settings.desi_db_host,
            port=settings.desi_db_port,
            database=settings.desi_db_database,
            user=settings.desi_db_username,
            password=settings.desi_db_password,
        )
        with conn:
            with conn.cursor() as cur:
                # TODO: Selecting this way is bad practice.
                # Write explicit selects for each table without any cleverness.
                cur.execute(f'SELECT {", ".join(columns)} FROM {table};')  # noqa: S608
                rows = np.asarray(cur.fetchall())
                data = load(rows=rows, columns=columns)
                save(settings.output_folder ,data, table=table)

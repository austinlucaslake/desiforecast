from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    desi_db_host: str = "replicator-db.desi.lbl.gov"
    desi_db_port: str = "5432"
    desi_db_database: str = "desi_dev"
    desi_db_username: str = "desi_reader"

    output_folder: Path = Path(__file__).parents[2] / "output"

    # NB: Secret
    desi_db_password: str

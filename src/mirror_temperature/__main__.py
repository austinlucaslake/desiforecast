from mirror_temperature.query_database import retrieve_and_store_data_from_database
from mirror_temperature.settings import Settings


def main():
    settings = Settings(_env_file=".env")
    retrieve_and_store_data_from_database(settings)


if __name__ == "__main__":
    main()

# mirror-temperature

This package contains code for predicting the temperature the DESI mirror should be set to in the morning, for it to have the correct temperature at night. 

# Installing
We are managing dependencies using `poetry`. To install poetry, please refer to their [documentation](https://python-poetry.org/docs/#installation). Once poetry has been isntalled, you can install the dependencies into a virtual environment by executing `poetry install` from this directory.

# Running
After installing the program, enter the associated virtual environment by running `poetry shell`. Then, the program can be run by calling `python -m mirror_temperature`.

In order to run, you need to supply a password for the database. The password can be provided by setting the environment variable `desi_db_password` - either directly into your environment, _or_ by saving it into a file called `.env` that should be kept in root.
NB: Do not commit the file called `.env`, as it contains secrets.

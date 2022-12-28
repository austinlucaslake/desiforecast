# desiforecast
Bayesian hierarchical model and machine learning of the DESI telemetry data.

## Prerequisites
- Anaconda
- Numpy
- Pandas
- Scipy
- Matplotlib
- Seaborn
- Psycopg 2
- Pytorch

## Querying Data
To query the telemetry data needed, run the following commands from the home directory of this repository: \
`cd data` \
`python query_telemetry_database.py` \
`cd ..`

## Data Visulization and Preprocessing (Work in Progress)
Data visualization and preprocessing can be found in the `data` directory under the respective folders.

## Running the Neural Network (Work in Progress)
To run the neural network and make temperature predictions for DESI instrument run the following command: \
`python neural_network.py`

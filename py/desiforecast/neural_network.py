"""This Python program computes a temperature prediction for the following day's temperature at Kitt's Peak, Arizona. A
Long Short-Term Memory Recurrant Neural Network is used to solve this regression problem. To account to prior
temperature distribution in an around the mountain, Monte Carlo dropout is used, which as an approximation for Bayesian
inference. This accounts for uncertainty in the estimates made by the neural network.

Notes
-----
To run this program use the following command from the directory containing this Python file and the data file
TODO: insert data file name
"""

import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib as mpl
from matplotlib import cm
mpl.style.use('tableau-colorblind10')
import matplotlib.pyplot as plt
import torch
from torch import nn, zeros, tensor
from torch.nn.functional import dropout
import argparse
parser = argparse.ArgumentParser(
                    prog = 'neural_network',
                    description = 'This program will ')
parser.add_argument('-t', '--train',
                    action='store_true',
                    help='Retrains the neural network')
parser.add_argument('-p', '--predict',
                    action='store_true',
                    help='Makes a prediction for the upcoming observing night')
args = parser.parse_args()

# global variables
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))
DATA_DIR = os.path.join(current_dir, 'data')
NB_DIR = os.path.join(current_dir, '..', '..', 'doc', 'nb')

# initalize data
class Data(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.from_numpy(X).to(device, dtype=torch.float32)
        self.y = torch.from_numpy(y).to(device, dtype=torch.float32)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

def load_data(data_path: str=os.path.join(DATA_DIR, 'neural_network_data.pkl'), device):
    """Loads training and test data
    
    Parameters
    ----------
    data_path : str
        Location to load training and testing data
        
    device : torch.device
        Loads computing device that will host neural network and tensor data
    
    Returns
    -------
    train_dataloader : Data
        Training telemetry data

    test_dataloader : Data
        Testing teleemetry data        
    """
    if os.isfile(data_path):
        X = pd.read_pickle()
        y = X.shift(1)
        y = y.iloc[1:]
        y = y['air_temp']
        X = X.iloc[:-1]
        input_size = X.shape[1]
        batch_size = 64
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)

        train_data = Data(X_train, y_train, device)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        test_data = Data(X_test, y_test, device)
        test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

class NeuralNetwork(nn.Module):
    """Initializes and trains an LSTM RNN to predict future temperatures for the DESI experiment.
    """
    def __init__(self, input_size, layer1_size:int=128, layer2_size:int=32, stack_size:int=2, dropout:float=0.05):
        super(self).__init__()
        self.hidden_layer1 = nn.LSTM(input_size, layer1_size, num_layers=stack_size, batch_first=True, dropout=dropout)
        self.hidden_layer2 = nn.LSTM(layer1_size, layer2_size, num_layers=stack_size, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(layer2_size, 1)
        # self.layer_stack = nn.Sequential(
        #     nn.LSTM(input_size, layer1_size, num_layers=stack_size, batch_first=True, dropout=dropout),
        #     nn.LSTM(layer1_size, layer2_size, num_layers=stack_size, batch_first=True, dropout=dropout),
        #     nn.Linear(self.layer2_size, 1)
        #
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, data: torch.tensor, training: bool):
        """Executes a forward pass of the neural network.
        
        Parameters
        ----------
        data : torch.tensor
            input data (temperature, seeing, etc.)

        Returns
        -------
        output : torch.tensor
            temperature estimation
        """
        data = torch.tensor(data, dtype=torch.float32)
        data, _ = self.hidden_layer1(data)
        # data = torch.tensor(data, dtype=torch.float32)
        data, _ = self.hidden_layer2(data)
        output = self.output_layer(data)
        return output
    
    def load_model(data_path: str=os.path.join(DATA_DIR, 'neural_network_model.pt')) -> None:
        """Loads the state dictionary of the neural network.

        Parameters
        ----------
        data_path : str
            Absolute path to load training and testing data            
        Returns
        -------
        None
        """
        self.load_state_dict(torch.load(data_path))
        self.eval()
    
    def train(self, train_dataloader: Data, epochs: int=100, output_path:str=os.path.join(save_dir, 'neural_network_loss.png')) -> None:
        """Trains the neural network.

        Parameters
        ----------
        train_dataloader : Data
            Input telemetry data
        
        epochs : int
            Number of epochs to train on
        
        output_path : str
            Absolute path to save output plot
            
        Returns
        -------
        None
        """
        global NB_DIR
        
        loss_values = []
        for epoch in trange(epochs, desc="Training neural network"):
            for X, y in train_dataloader:
                self.optimizer.zero_grad()
                prediction = self(X, training=True)
                loss = self.loss(prediction.reshape(-1), y)
                loss_values.append(loss.item())
                loss.backward()
                self.optimizer.step()
        
        steps = np.linspace(0, epochs, len(loss_values))
        
        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(steps, np.array(loss_values))
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(output_path)
        plt.show()
        plt.close(fig)
        
        self.save_model()
        return loss_values
        
    def save_model(self, output_path:str=os.path.join(DATA_DIR, 'neural_network_model.pt')) -> None:
        """Saves the state dictionary of the neural network.

        Parameters
        ----------
        output_path : str
           Absolute path of file to hold serialized neural network model.
            
        Returns
        -------
        None
        """
        torch.save(self.state_dict(), output_path)
    
    def predict(self, test_dataloader: Data, output_path:str=os.path.join(save_dir, 'neural_network_loss.png')) -> None:
        """Makes predictions using the trained neural network.

        Parameters
        ----------
        test_dataloader : Data
            Input telemetry data
        
        output_path : str
            Absolute path to save output plot
        
        Returns
        -------
        loss_values : list[float]
            Test data loss values
        """
        loss_values = []
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc="Making temperature predictions"):
                outputs = self(X, training=False)
                predicted = np.where(outputs < 0.5, 0, 1)
                predicted = list(itertools.chain(*predicted)) 
                loss = self.loss(predicted, y.unsqueeze(-1))
                loss_values.append(loss)
    
        steps = np.linspace(0, epochs, len(loss_list))
        
        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(steps, np.array(loss_values))
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(output_path)
        plt.show()
        plt.close(fig)
        
        self.save()
        return loss_values

def main():
    """Main program.

    Parameters
    ----------
    None

    Returns
    -------
    main : int
        Exit status that will be passed to ``sys.exit()``.
    """
    global DEVICE
    
    train_dataloader, test_dataloader = load_data(DEVICE)
    model = NeuralNetwork(input_size=input_size).to(DEVICE)
    
    today = datetime.now(timezone.utc)
    tomorrow = (today+timedelta(days=1))
    
    if args.train:
        train_loss = model.train(train_dataloader)
    if args.predict:
        test_loss = model.predict(test_dataloader)

    return 0

if __name__ == '__main__':
    sys.exit(main())

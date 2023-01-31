"""

This Python program computes a temperature prediction for the following day's temperature at Kitt's Peak, Arizona. A
Long Short-Term Memory Recurrant Neural Network is used to solve this regression problem. To account to prior
temperature distribution in an around the mountain, Monte Carlo dropout is used, which as an approximation for Bayesian
inference. This accounts for uncertainty in the estimates made by the neural network.

Notes
-----
To run this program use the following command from the directory containing this Python file and the data file
TODO: insert data file name
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import torch
from torch import nn, zeros, tensor
from torch.nn.functional import dropout
from torch.autograd import Variable

class BayesianLSTM(nn.Module):
    """
    This class initializes and trains an LSTM RNN to predict future temperatures for the DESI experiment
    """
    def __init__(self, input_size):
        super(BayesianLSTM, self).__init__()
        self.input_size = input_size
        self.layer1_size = 128
        self.layer2_size = 32
        self.stack_size = 2
        self.dropout_prob = 0.05
        self.hidden_layer1 = nn.LSTM(self.input_size, self.layer1_size, num_layers=self.stack_size, batch_first=True)
        self.hidden_layer2 = nn.LSTM(self.layer1_size, self.layer2_size, num_layers=self.stack_size, batch_first=True)
        self.output_layer = nn.Linear(self.layer2_size, 1)

        self.loss_function = nn.MSELoss

    def forward(self, data):
        """
        This function will implement a forward pass of the neural network.
        
        Parameters
        ----------
        data: np.ndarray
            input data (temperature, seeing, etc.)

        Returns
        -------
        output: torch.tensor
            temperature estimation
        """

        batch_size = data.size()
        layer1_state = self.init_tensors(batch_size, self.layer1_size)
        layer1_output, _ = self.hidden_layer1(data, layer1_state)
        layer1_output = dropout(layer1_output, p=self.dropout_probability, training=True)
        layer2_state = self.init_tensors(batch_size, self.layer2_size)
        layer2_output, _ = self.hidden_layer2(batch_size, layer2_state)
        layer2_output = dropout(layer1_output, p=self.dropout_probability, training=True)
        output = self.output_layer(layer2_output[:, -1, :])
        return output

    def init_tensors(self, batch_size, layer_size):
        """
        This function will initialize the hidden layer tensors as needed throughout a forward pass.
        
        Parameters
        ----------
        data: np.ndarray
            input data (temperature, seeing, etc.)

        Returns
        -------
        output: torch.tensor
            temperature estimation
        """

        hidden = Variable(zeros(self.stack_size, batch_size, layer_size))
        cell = Variable(zeros(self.stack_size, batch_size, layer_size))
        return hidden, cell

    def loss(self, prediction, true_val):
        """
        This function calculates the loss from a forward pass of the neural network.

        Parameters
        ----------
        prediction: np.ndarray
            temperature estimation

        Returns
        -------
        loss: torch.tensor
            mean squared error
        """

        loss = self.loss_function(prediction, true_val)
        return loss

    def predict(self, data):
        """
        This functions will output a prediction
        
        Parameters
        ----------
        data: np.ndarray
            input data (temperature, seeing, etc.)

        Returns
        -------
        prediction: np.ndarray
            temperature estimation
        """
        
        prediction = self(tensor(data, dtype=torch.float32)).view(-1).detach().numpy()
        return prediction


if __name__ == '__main__':
    today = datetime.now(timezone.utc)
    tomorrow = (today+timedelta(days=1))
    verbose = True
    test = True
    if verbose:
        print(f'Python {sys.version}')
        print(today.strftime("%B %d, %Y"))

    # TODO: preprocess data
    data = ['hello world']
    model = BayesianLSTM(input_size=len(data))
    if test:
        pass
    else:
        # print(model.predict(tomorrow))
        pass

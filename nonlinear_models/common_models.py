# Code from this file is modified from:
# https://github.com/pliang279/PID/blob/main/unimodals/common_models.py

import torch 
from torch import nn 
from torch.nn import functional as F 

class Linear(torch.nn.Module): 
    """Linear layer with 0 bias"""

    def __init__(self, indim, outdim): 
        """Initialize Linear Layer
        Args: 
            indim (int): input dimension 
            outdim (int): output dimension 
        """
        super(Linear, self).__init__() 
        self.fc = nn.Linear(indim, outdim)
    
    def forward(self, x): 
        """Apply linear layer to input. 
        Args: 
            x (torch.Tensor): input tensor 
        Returns:
            torch.Tensor: output tensor 
        """
        return self.fc(x) 
    

class MLP(torch.nn.Module): 
    """Two layered perceptron"""

    def __init__(self, indim, hiddim, outdim=1, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron
        Args:
            indim (int): input dimension 
            hiddim (int): hidden layer dimention 
            outdim (int, optional): output dimension. Default is 1. 
            dropout (bool, optional): whether to apply dropout or not. Default is false.
            dropoutp (float, optional): dropout probability. Default is 0.1. 
            output_each_layer (bool, optional): whether to return output of each layer as a list. Default is false.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)
    
    def forward(self, x): 
        """Apply MLP to Input.
        Args:
            x (torch.Tensor): Layer Input
        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output2)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2
    

class Identity(nn.Module):
    """Identity Module."""
    
    def __init__(self):
        """Initialize Identity Module."""
        super().__init__()

    def forward(self, x):
        """Apply Identity to Input.
        Args:
            x (torch.Tensor): Layer Input
        Returns:
            torch.Tensor: Layer Output
        """
        return x
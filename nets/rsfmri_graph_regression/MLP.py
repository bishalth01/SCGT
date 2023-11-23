import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np
from nets.rsfmri_graph_regression.SANGRU import GruKRegion

from nets.rsfmri_graph_regression.tcn import TemporalConvNet
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import global_mean_pool

# Define the MLP model with 300 input features
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(373, 2048)  # 300 input features, 8 neurons in the hidden layer
        self.output_layer = nn.Linear(2048, 1)  # 1 output neuron for regression
        self.readout="mean"

    def forward(self, g, h):
        
        num_nodes = 100

        batch_main = int(h.shape[0]/num_nodes)
        reshaped_data = h.reshape(batch_main * num_nodes, h.shape[-1])

        h = torch.relu(self.hidden_layer(reshaped_data))

        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        #x = torch.relu(self.hidden_layer(x))  # Apply ReLU activation to the hidden layer
        x = self.output_layer(hg)
        return x
    
    def loss(self, scores, targets):

        loss = nn.SmoothL1Loss()(scores, targets)
        
        return loss
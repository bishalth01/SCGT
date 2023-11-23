import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
import math
from torch_sparse import spspmm
import numpy as np
import torch_geometric.utils as pyg_utils
from torch_scatter import scatter_mean, scatter_add
from einops import rearrange, repeat
from torch_geometric.nn.aggr.gmt import GraphMultisetTransformer


class ModuleSERO(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, batch, node_axis=1):
        # assumes shape [... x node x ... x feature]
        #x = x.reshape(64,-1,x.shape[-1])
        x_readout = gap(x, batch)
        #x_readout = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(1))).mean(node_axis), x_graphattention
    

class ModuleGARO(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, batch_main, ratio=0,  node_axis=-2):
        if ratio>0:
            nodes_to_keep = int (batch_main * 100 * ratio)
            x = x[0:nodes_to_keep]
        x = x.reshape(batch_main,-1,x.shape[-1])
        x_q = self.embed_query(x.mean(node_axis))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q.unsqueeze(1), x_k.transpose(2,1))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.transpose(2,1))).mean(node_axis), x_graphattention

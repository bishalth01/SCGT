import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling
import torch.nn.init as init
import dgl
import numpy as np
from nets.rsfmri_graph_regression.SANGRU import GruKRegion

from nets.rsfmri_graph_regression.attention_readouts import ModuleGARO



"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.node_graph_transformer_layer import NodeGraphTransformerLayer
from layers.node_community_graph_transformer_layer import NodeCommunityGraphTransformerLayer
#from layers.node_graph_transformer_layer import NodeGraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAN_NodeLPEWithPosEncCommunityTopK(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        num_atom_type = 360
        num_bond_type=1

        self.window_size= 18
        self.num_gru_layers = 4
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        
        LPE_layers = net_params['LPE_layers']
        LPE_dim = net_params['LPE_dim']
        LPE_n_heads = net_params['LPE_n_heads']
        
        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.dropout = dropout

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        ratio = net_params['ratio']
        self.ratio = ratio

        self.num_fc_layers= net_params['num_fc_layers']


        self.extract = GruKRegion(
                    out_size=GT_hidden_dim, kernel_size=self.window_size,
                    layers=self.num_gru_layers, dropout=dropout)
        
        self.embedding_e = nn.Linear(1, GT_hidden_dim)
        self.linear_A = nn.Linear(2, LPE_dim)


        self.embedding_lap_pos_enc = nn.Linear(LPE_dim, GT_hidden_dim)
        
        self.layers = nn.ModuleList([ NodeCommunityGraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, share_weights=False) for _ in range(GT_layers-1) ])
        
        self.layers.append(NodeCommunityGraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, share_weights=False))   

        self.topkpool = TopKPooling(GT_hidden_dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)  


        if self.readout=="garo":
            self.garoreadout = ModuleGARO(GT_hidden_dim, GT_hidden_dim, dropout=dropout, upscale=1.0)

        #self.fc_dim1 = GT_hidden_dim
        self.fc_dim1 = net_params['fc_dim1']
        self.fc_dim2 = net_params['fc_dim2']

        if self.num_fc_layers==2:
            self.fc1 = torch.nn.Linear(GT_hidden_dim, self.fc_dim1)
            self.fc2 = torch.nn.Linear(self.fc_dim1, 1)
        elif self.num_fc_layers==3:
            self.fc1 = torch.nn.Linear(GT_hidden_dim, self.fc_dim1)
            self.fc2 = torch.nn.Linear(self.fc_dim1, self.fc_dim2)
            self.fc3 = torch.nn.Linear(self.fc_dim2, 1)
        else:
            self.fc1 = torch.nn.Linear(GT_hidden_dim, 1)
        
   
    def forward(self, g, h, e,EigVecs, EigVals, pseudo):

        num_nodes = 100

        h = h.reshape(-1, 100, 373)
        h = h[:,:,:360]

        h = self.extract(h)
        e = self.embedding_e(e)  


        batch_main = int(h.shape[0])

        h = h.reshape(batch_main*num_nodes, h.shape[-1])

        h_lap_pos_enc = self.embedding_lap_pos_enc(EigVecs.float()) 
        h = h + h_lap_pos_enc
        
        h = self.in_feat_dropout(h)
        
        

        # GNN
        for conv in self.layers:
            h, e = conv(g, h, e, EigVecs)

        g.ndata['h'] = h
        

        #Pooling

        batch = torch.zeros(h.size(0), dtype=torch.long, device='cuda')

        h, edge_index, edge_attr, batch, perm, score = self.topkpool(h, (g.edges()[0], g.edges()[1]), g.edata['feat'], batch)

        score= 0

        nodes_to_keep = int (batch_main * 100 * 1)
        h = h[0:nodes_to_keep]
        h = h.reshape(batch_main,-1,h.shape[-1])

        if self.readout == "garo":
            hg, _ = self.garoreadout(h, batch_main, self.ratio)
        else:
            hg = torch.mean(h, dim=1)

        if self.num_fc_layers==2:
            x = F.relu(self.fc1(hg))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.fc2(x))
        elif self.num_fc_layers==3:
            x = F.relu(self.fc1(hg))
            x = F.dropout(x, p=self.dropout)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.relu(self.fc3(x))
        else:
            x = torch.relu(self.fc1(hg))
            
        return x, score
        
    def loss(self, scores, targets):

        loss = nn.MSELoss()(scores, targets)
        
        return loss
    



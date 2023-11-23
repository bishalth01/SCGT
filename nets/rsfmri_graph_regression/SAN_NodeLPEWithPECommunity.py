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

class SAN_NodeLPEWithPosEncCommunity(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        #num_atom_type = net_params['num_atom_type']
        #num_bond_type = net_params['num_bond_type']

        # num_atom_type = 373
        # num_bond_type=1

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
        
        #self.embedding_h = nn.Embedding(num_atom_type, GT_hidden_dim-LPE_dim)#Remove some embedding dimensions to make room for concatenating laplace encoding
        # self.embedding_h = nn.Linear(num_atom_type, GT_hidden_dim-LPE_dim )
        # self.embedding_e = nn.Linear(1, GT_hidden_dim)

        self.extract = GruKRegion(
                    out_size=GT_hidden_dim, kernel_size=self.window_size,
                    layers=self.num_gru_layers, dropout=dropout)
        
        self.embedding_e = nn.Linear(1, GT_hidden_dim)
        self.linear_A = nn.Linear(2, LPE_dim)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        # self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)

        self.embedding_lap_pos_enc = nn.Linear(LPE_dim, GT_hidden_dim)
        
        self.layers = nn.ModuleList([ NodeCommunityGraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, share_weights=False) for _ in range(GT_layers-1) ])
        
        self.layers.append(NodeCommunityGraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, share_weights=False))   

        # self.topkpool = TopKPooling(GT_hidden_dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)  

        # Apply Xavier initialization to the weights of the TopKPooling layer
        # for param in self.topkpool.parameters():
        #     if len(param.shape) > 1:
        #         init.xavier_uniform_(param)

        if self.readout=="garo":
            self.garoreadout = ModuleGARO(GT_hidden_dim, GT_hidden_dim, dropout=dropout, upscale=1.0)

        #self.fc_dim1 = GT_hidden_dim
        self.fc_dim1 = net_params['fc_dim1']
        self.fc_dim2 = net_params['fc_dim2']

        #self.MLP_layer = MLPReadout(GT_hidden_dim, 1)   # 1 out dim since regression problem   

        # self.fc1 = torch.nn.Linear(GT_hidden_dim, self.fc_dim2)
        # self.bn1 = torch.nn.BatchNorm1d(self.fc_dim2)
        # # self.fc2 = torch.nn.Linear(self.fc_dim1, self.fc_dim2)
        # # self.bn2 = torch.nn.BatchNorm1d(self.fc_dim2)
        # self.fc3 = torch.nn.Linear(self.fc_dim2, 1)

        if self.num_fc_layers==2:
            self.fc1 = torch.nn.Linear(GT_hidden_dim, self.fc_dim1)
            #if self.batch_norm:
            # self.bn1 = torch.nn.BatchNorm1d(self.fc_dim1)
            self.fc2 = torch.nn.Linear(self.fc_dim1, 1)
        elif self.num_fc_layers==3:
            self.fc1 = torch.nn.Linear(GT_hidden_dim, self.fc_dim1)
            #if self.batch_norm:
            # self.bn1 = torch.nn.BatchNorm1d(self.fc_dim1)
            self.fc2 = torch.nn.Linear(self.fc_dim1, self.fc_dim2)
            #if self.batch_norm:
            # self.bn2 = torch.nn.BatchNorm1d(self.fc_dim2)
            self.fc3 = torch.nn.Linear(self.fc_dim2, 1)
        else:
            self.fc1 = torch.nn.Linear(GT_hidden_dim, 1)
        
        
    # def forwardss(self, g, h, e, EigVecs, EigVals, pseudo):

    #     num_bins = 20
    #     num_nodes = 100
    #     num_time_points = 373

    #     h = self.embedding_h(h)
    #     e = self.embedding_e(e)  

    #     PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
    #     empty_mask = torch.isnan(PosEnc) # (Num nodes) x (Num Eigenvectors) x 2
        
    #     PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
    #     PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
    #     PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
        
        
    #     #1st Transformer: Learned PE
    #     PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
        
    #     #remove masked sequences
    #     PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
        
    #     #Sum pooling
    #     PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        
    #     #Concatenate learned PE to input embedding
    #     h = torch.cat((h, PosEnc), 1)


    #     #print("Mean of features is" + str(torch.mean(h)))
    #     batch_main = int(h.shape[0]/num_nodes)
    #     #reshaped_data = h.reshape(batch_main * num_nodes, h.shape[-1])
        
    #     # input embedding
    #     # h = self.embedding_h(reshaped_data)
    #     # e = self.embedding_e(e)  
        
    #     h = self.in_feat_dropout(h)
        
    #     h = h.reshape(batch_main*num_nodes, h.shape[-1])

    #     # GNN
    #     for conv in self.layers:
    #         h, e = conv(g, h, e, pseudo)

    #     g.ndata['h'] = h
        
    #     # # input embedding
    #     # h = self.embedding_h(h)
    #     # e = self.embedding_e(e)  
        

    #     # #Concatenate learned PE to input embedding
    #     # h = torch.cat((h, PosEnc), 1)
        
    #     # h = self.in_feat_dropout(h)
        
        
    #     # # GNN
    #     # for conv in self.layers:
    #     #     h, e = conv(g, h, e)
    #     # g.ndata['h'] = h
        
    #     # if self.readout == "sum":
    #     #     hg = dgl.sum_nodes(g, 'h')
    #     # elif self.readout == "max":
    #     #     hg = dgl.max_nodes(g, 'h')
    #     # elif self.readout == "mean":
    #     #     hg = dgl.mean_nodes(g, 'h')

    #     # else:
    #     #     hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

    #     hg, _ = self.garoreadout(h)
             
    #     return self.MLP_layer(hg)


    #def forward(self, g, h, e, EigVecs, EigVals, pseudo):
    def forward(self, g, h, e,EigVecs, EigVals, pseudo):

        num_bins = 20
        num_nodes = 100
        num_time_points = 373

        h = h.reshape(-1, 100, 373)
        h = h[:,:,:360]

        h = self.extract(h)
        e = self.embedding_e(e)  
        

        # PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
        # empty_mask = torch.isnan(PosEnc) # (Num nodes) x (Num Eigenvectors) x 2
        
        # PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        # PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
        # PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
        
        
        # #1st Transformer: Learned PE
        # PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
        
        # #remove masked sequences
        # PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
        
        # #Sum pooling
        # PosEnc = torch.nansum(PosEnc, 0, keepdim=False)

        # PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
        # empty_mask = torch.isnan(PosEnc) # (Num nodes) x (Num Eigenvectors) x 2
        
        # PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        # PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
        # PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
        
        
        # #1st Transformer: Learned PE
        # PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
        
        #remove masked sequences
        # PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
        
        #Sum pooling
        # PosEnc = torch.nansum(PosEnc, 0, keepdim=False)

        batch_main = int(h.shape[0])

        h = h.reshape(batch_main*num_nodes, h.shape[-1])

        h_lap_pos_enc = self.embedding_lap_pos_enc(EigVecs.float()) 
        h = h + h_lap_pos_enc
        
        #Concatenate learned PE to input embedding
        # h = torch.cat((h, PosEnc), 1)


        #print("Mean of features is" + str(torch.mean(h)))
        
        reshaped_data = h.reshape(batch_main * num_nodes, h.shape[-1])
        
        # # input embedding
        # h = self.embedding_h(reshaped_data)
        # e = self.embedding_e(e)  
        
        h = self.in_feat_dropout(h)
        
        

        # GNN
        for conv in self.layers:
            h, e = conv(g, h, e, EigVecs)

        g.ndata['h'] = h
        
        # # input embedding
        # h = self.embedding_h(h)
        # e = self.embedding_e(e)  
        

        # #Concatenate learned PE to input embedding
        # h = torch.cat((h, PosEnc), 1)
        
        # h = self.in_feat_dropout(h)
        
        
        # # GNN
        # for conv in self.layers:
        #     h, e = conv(g, h, e)
        # g.ndata['h'] = h

        #Pooling

        # batch = torch.zeros(h.size(0), dtype=torch.long, device='cuda')

        # h, edge_index, edge_attr, batch, perm, score = self.topkpool(h, (g.edges()[0], g.edges()[1]), g.edata['feat'], batch)

        # #h, edge_index, edge_attr, batch, perm, score = self.topkpool(h, g.edges()[0], g.edata['feat'])
        # pos = pseudo[perm]

        
        # hg, _ = self.garoreadout(h, batch_main, self.ratio)

        score= 0

        # nodes_to_keep = int (batch_main * 100 * 1)
        # h = h[0:nodes_to_keep]

        h = h.reshape(batch_main,-1,h.shape[-1])

        if self.readout == "garo":
            hg, _ = self.garoreadout(h, batch_main)
        else:
            hg = torch.mean(h, dim=1)



        # x = self.bn1(F.relu(self.fc1(hg)))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.bn2(F.relu(self.fc2(x)))
        # x= F.dropout(x, p=self.dropout, training=self.training)
        # x = torch.relu(self.fc3(x))

        # x = F.relu(self.fc1(hg))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # # x = F.relu(self.fc2(x))
        # # x= F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.fc3(x))

        if self.num_fc_layers==2:
            x = F.relu(self.fc1(hg))
            # if self.batch_norm:
            #     x = self.bn1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            #x = torch.sigmoid(self.fc2(x))
            x = F.relu(self.fc2(x))
            # if self.batch_norm:
            #     x = self.bn2(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.num_fc_layers==3:
            x = F.relu(self.fc1(hg))
            # if self.batch_norm:
            #     x = self.bn1(x)
            x = F.dropout(x, p=self.dropout)
            x = F.relu(self.fc2(x))
            # if self.batch_norm:
            #     x = self.bn2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            #x = torch.sigmoid(self.fc3(x))
            x = torch.relu(self.fc3(x))
        else:
            x = torch.relu(self.fc1(hg))
            
        # return self.MLP_layer(hg)
        return x, score
        
    def loss(self, scores, targets):

        #loss = nn.L1Loss()(scores, targets)
        loss = nn.MSELoss()(scores, targets)
        
        return loss
    



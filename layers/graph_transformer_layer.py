import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func


def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L*torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func



"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias):
        super().__init__()
        
       
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph=full_graph
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            
            if self.full_graph:
                self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            
            if self.full_graph:
                self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        #nn.init.xavier_uniform_(self.O_h.weight)

    def propagate_attention(self, g):
            # Compute attention score
            g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
            g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

            # Send weighted values to target nodes
            eids = g.edges()
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    

    def forward(self, g, h):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        #E = self.E(e)
        
        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            #E_2 = self.E_2(e)
            
        V_h = self.V(h)

        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        # g.ndata['Q_h'] = Q_h.view(-1, Q_h.shape[1], self.num_heads, self.out_dim)
        # g.ndata['K_h'] = K_h.view(-1, K_h.shape[1], self.num_heads, self.out_dim)

        g.ndata['Q_h'] = Q_h.view(-1,  self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1,  self.num_heads, self.out_dim)
        #g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)
        
        
        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            #g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        
        return h_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads, full_graph, use_bias)

        # Inside MultiHeadAttentionLayer and GraphTransformerLayer __init__ method




        
        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            #self.batch_norm1_h = nn.BatchNorm1d(360)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
        
        # For GraphTransformerLayer FFN layers
        nn.init.xavier_uniform_(self.FFN_h_layer1.weight)
        nn.init.xavier_uniform_(self.FFN_h_layer2.weight)

    def forward(self, g, h):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        h_attn_out = self.attention(g, h)
        
        #Concat multi-head outputs
        #h = h_attn_out.view(-1, h.shape[1], self.out_channels)
        h = h_attn_out.view(-1, self.out_channels)
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)         

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
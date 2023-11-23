'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import os
import glob
import h5py
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
#from imports.gdc import GDC
from scipy.linalg import eigh
import hashlib
from nilearn.connectome import ConnectivityMeasure
import dgl
from scipy import sparse as sp

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])



    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice
    if data.graphs is not None:
        slices['graphs'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    graphs=[]

    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))
        graphs.append(res[j][5])

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)

    y_arr = np.stack(y_list)

    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr)#.long()  # prediction
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch, graphs=graphs )


    data, slices = split(data, batch_torch)

    return data, slices


def compute_laplacian_eigenvectors(adjacency_matrix, num_eigenvectors=10):
    # Compute the Laplacian matrix
    np.fill_diagonal(adjacency_matrix, 0)
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Compute the eigenvectors of the Laplacian matrix
    _, eigenvectors = eigh(laplacian_matrix)

    #Sort the eigenvectors
    idx = np.argsort(np.real(_))
    eigenvectors = np.real(eigenvectors[:, idx])
    eigenvectors = eigenvectors[:, 1:num_eigenvectors+1]

    return eigenvectors.astype(np.float32)


def compute_wl_positional_encoding(G, max_iter=2):
    """
    WL-based absolute positional embedding
    adapted from "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
    Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
    https://github.com/jwzhanggy/Graph-Bert
    """

    node_color_dict = {}
    node_neighbor_dict = {}

    A = nx.to_scipy_sparse_matrix(G)  # Compute the adjacency matrix

    edge_list = A.nonzero()
    node_list = list(G.nodes())

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for u1, u2 in zip(edge_list[0], edge_list[1]):
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    for node, color_enc in node_color_dict.items():
        G.nodes[node]['wl_pos_enc'] = color_enc

    return G

def normalise_timeseries(timeseries: np.ndarray) -> np.ndarray:
    """
    :param normalisation:
    :param timeseries: In  format TS x N
    :return:
    """
    normalisation="subject_norm"
    if normalisation == "subject_norm":
        flatten_timeseries = timeseries.flatten().reshape(-1, 1)
        scaler = RobustScaler().fit(flatten_timeseries)
        timeseries = scaler.transform(flatten_timeseries).reshape(timeseries.shape).T
    else:  # No normalisation
        timeseries = timeseries.T

    return timeseries


def threshold_adj_array(adj_array: np.ndarray, threshold: int, num_nodes: int) -> np.ndarray:
    num_to_filter: int = int((threshold / 100.0) * (num_nodes * (num_nodes - 1) / 2))

    # For threshold operations, zero out lower triangle (including diagonal)
    adj_array[np.tril_indices(num_nodes)] = 0

    # Following code is similar to bctpy
    indices = np.where(adj_array)
    sorted_indices = np.argsort(adj_array[indices])[::-1]
    adj_array[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0

    # Just to get a symmetrical matrix
    adj_array = adj_array + adj_array.T

    # Diagonals need connection of 1 for graph operations
    adj_array[np.diag_indices(num_nodes)] = 1.0

    return adj_array

def create_thresholded_graph(adj_array: np.ndarray, threshold: int, num_nodes: int) -> nx.DiGraph:
    adj_array = threshold_adj_array(adj_array, threshold, num_nodes)

    return nx.from_numpy_array(adj_array, create_using=nx.DiGraph)





def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g



#----------------------------SAN Positional Encoding----------------------------------------------

# def laplace_decomp(g, max_freqs):


#     # Laplacian
#     n = g.number_of_nodes()
#     A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
#     N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
#     L = sp.eye(g.number_of_nodes()) - N * A * N

#     # Eigenvectors with numpy
#     EigVals, EigVecs = np.linalg.eigh(L.toarray())
#     EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

#     # Normalize and pad EigenVectors
#     EigVecs = torch.from_numpy(EigVecs).float()
#     EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
#     if n<max_freqs:
#         g.ndata['lap_pos_enc'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
#     else:
#         g.ndata['lap_pos_enc']= EigVecs
        
    
#     #Save eigenvales and pad
#     EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
#     if n<max_freqs:
#         EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
#     else:
#         EigVals=EigVals.unsqueeze(0)
        
    
#     #Save EigVals node features
#     g.ndata['lap_pos_enc'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
#     return g

def laplace_decomp(g, max_freqs):


    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    return g

def add_edge_laplace_feats(g):

    
    EigVals = g.ndata['EigVals'][0].flatten()
    
    source, dest = g.find_edges(g.edges(form='eid'))
    
    #Compute diffusion distances and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs']-g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(),1).unsqueeze(2)
    
    
    #No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']
    
    return g

def avergage_sliding_timepoints(time_series, num_nodes):
        sliding_window = 50
        stride=3
        dynamic_length = 373
        num_sliding_windows = int((dynamic_length - sliding_window) / stride) + 1

        averaged_data = np.zeros((num_nodes, num_sliding_windows))

        for i in range(num_sliding_windows):
            start = i * stride
            end = start + sliding_window
            window_data = time_series[:, start:end]
            averaged_data[:, i] = np.mean(window_data, axis=1)
        
        return averaged_data

def read_sigle_data(data_dir,filename,use_gdc =False):

    num_nodes=100

    temp = dd.io.load(osp.join(data_dir, filename))

    ts = temp['corr']

    # ts = ts.squeeze(0)
    
    #ts = avergage_sliding_timepoints(ts.T, 100).T

    conn_measure = ConnectivityMeasure(
                        kind='correlation',
                        vectorize=False)
    corr_arr = conn_measure.fit_transform([ts])
    assert corr_arr.shape == (1, num_nodes, num_nodes)

    # read edge and edge attribute
    #pcorr = np.abs(temp['pcorr'][()]).reshape(53,53)
    pcorr = np.abs(corr_arr).reshape(num_nodes,num_nodes)

    num_nodes = pcorr.shape[0] #100 nodes
    G = from_numpy_matrix(pcorr) #Matrix
    # corr_arr = corr_arr.reshape(num_nodes,num_nodes)
    # G = create_thresholded_graph(corr_arr, threshold=30, num_nodes=num_nodes)
    A = nx.to_scipy_sparse_matrix(G) #Adjacency matrix
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    #att = normalise_timeseries(temp['corr'][()]).T #Attributes
    #att = normalise_timeseries(temp['corr']) #Attributes
    #att = temp['corr'].T
    att = ts.T
    att = normalise_timeseries(att).T
    label = temp['label'][()] #Labels

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label))#.long()  # prediction
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes = num_nodes)
    #dgl_graph = laplacian_positional_encoding(dgl_graph, 10)
    dgl_graph = laplace_decomp(dgl_graph, 10)

    #Compute the laplacian eigenvectors

    #pos_emb_torch = torch.from_numpy(compute_laplacian_eigenvectors(pcorr))
    # Concatenate the positional embedding to the node features
    #att_torch = torch.cat([att_torch, pos_emb_torch], dim=1)
    

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att, graph =dgl_graph )

    # if use_gdc:
    #     '''
    #     Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
    #     '''
    #     data.edge_attr = data.edge_attr.squeeze()
    #     gdc = GDC(self_loop_weight=1, normalization_in='sym',
    #               normalization_out='col',
    #               diffusion_kwargs=dict(method='ppr', alpha=0.2),
    #               sparsification_kwargs=dict(method='topk', k=30,
    #                                          dim=0), exact=True)
    #     data = gdc(data)
    #     return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    # else:
        #return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes
    return edge_att.data.numpy(),edge_index.data.numpy(),att_torch,label,num_nodes, dgl_graph

if __name__ == "__main__":
    data_dir = '/home/azureuser/projects/BrainGNN/data/ABIDE_pcp/cpac/filt_noglobal/raw'
    filename = '50346.h5'
    read_sigle_data(data_dir, filename)







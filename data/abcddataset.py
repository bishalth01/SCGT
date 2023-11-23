import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl
import torch.nn.functional as F


from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib
from data.ABIDEDatasetPECryst import ABIDEDatasetPECryst
from data.ABIDEDatasetPEFluid import ABIDEDatasetPEFluid
from data.ABIDEDatasetPESex import ABIDEDatasetPESex
from data.ABIDEDatasetPETotalComp import ABIDEDatasetPETotalComp
from pyts.approximation import SymbolicAggregateApproximation
from utils.utils import train_val_test_split


class ABCDDGL(torch.utils.data.Dataset):
    def __init__(self, dataset, num_graphs=None):
        self.dataset = dataset
        self.num_graphs = num_graphs
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(dataset)
        self._prepare()         
    
    def _prepare(self):
        #print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        print("preparing graphs for ABCD with graphs = " + str(self.num_graphs) )
        
        # for molecule in self.data:
        #     node_features = molecule['atom_type'].long()
            
        #     adj = molecule['bond_type']
        #     edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
        #     edge_idxs_in_adj = edge_list.split(1, dim=1)
        #     edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
        #     # Create the DGL Graph
        #     g = dgl.DGLGraph()
        #     g.add_nodes(molecule['num_atom'])
        #     g.ndata['feat'] = node_features
            
        #     for src, dst in edge_list:
        #         g.add_edges(src.item(), dst.item())
        #     g.edata['feat'] = edge_features
            
        #     self.graph_lists.append(g)
        #     self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

        
        for i in range(self.n_samples):
            molecule = self.dataset[i]
            node_features = molecule.x
            
            edge_list = molecule.edge_index.t().long()  # Assuming edge_index is in (2, num_edges) format
            edge_features = molecule.edge_attr
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule.num_nodes)
            g.ndata['feat'] = node_features
            
            src, dst = edge_list[:, 0], edge_list[:, 1]
            g.add_edges(src, dst)
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule.y)
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class ABCDDatasetDGL(torch.utils.data.Dataset):
    # def __init__(self, name='Zinc'):
    #     t0 = time.time()
    #     self.name = name
        
    #     self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
    #     self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
    #     data_dir='./data/molecules'
        
    #     if self.name == 'ZINC-full':
    #         data_dir='./data/molecules/zinc_full'
    #         self.train = ABCDDGL(data_dir, 'train', num_graphs=220011)
    #         self.val = ABCDDGL(data_dir, 'val', num_graphs=24445)
    #         self.test = ABCDDGL(data_dir, 'test', num_graphs=5000)
    #     else:            
    #         self.train = ABCDDGL(data_dir, 'train', num_graphs=10000)
    #         self.val = ABCDDGL(data_dir, 'val', num_graphs=1000)
    #         self.test = ABCDDGL(data_dir, 'test', num_graphs=1000)
    #     print("Time taken: {:.4f}s".format(time.time()-t0))

        def __init__(self, dataset_name='ABCD'):
            t0 = time.time()
            self.dataset_name = dataset_name
            self.num_atom_type = 100  # Update this with your actual number of atom types
            self.num_bond_type = 1  # Update this with your actual number of bond types
            
            # Load your dataset here using the dataset_name
            
            self.train = ABCDDGL(self.train_dataset, num_graphs=10000)  # Update with the actual train dataset
            self.val = ABCDDGL(self.val_dataset, num_graphs=1000)      # Update with the actual validation dataset
            self.test = ABCDDGL(self.test_dataset, num_graphs=1000)    # Update with the actual test dataset
            
            print("Time taken: {:.4f}s".format(time.time() - t0))



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



def make_full_graph(g):

    
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
    return full_g


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



class ABCDDataset(torch.utils.data.Dataset):

    def __init__(self, name, save_path, intelligence):
        self.name = name
        self.save_path = save_path

        start = time.time()
        data_path = "/data/users2/bthapaliya/PEIntelligenceABCDPredictionBrainGNN/data/data/Output"
        #data_path = "/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN/data/data/Output"

        if intelligence == "cryst":
            dataset = ABIDEDatasetPECryst(data_path, "ABCD")
        elif intelligence== "fluid":
            dataset = ABIDEDatasetPEFluid(data_path, "ABCD")
        elif intelligence=="sex":
            dataset = ABIDEDatasetPESex(data_path, "ABCD")
        else:
            dataset = ABIDEDatasetPETotalComp(data_path, "ABCD")


        #Normalizing inputs
        dataset.data.x[dataset.data.x == float('inf')] = 0

        min_values, _ = torch.min(dataset.data.x, dim=1)
        max_values, _ = torch.max(dataset.data.x, dim=1)

        # dataset.data.x = (dataset.data.x - min_values.view(-1,1)) / (max_values.view(-1,1) - min_values.view(-1,1))

        #Removing outliers and normalizing output labels

        # Assuming min_indices contains the indices of the top 5 smallest values in dataset.data.y
        #----------------------------
        # min_values, min_indices = torch.topk(dataset.data.y.squeeze(), k=5, largest=False)
        # indices_to_remove = min_indices.tolist()

        # dataset.data.x = dataset.data.x.reshape(int(dataset.data.x.shape[0]/100), 100, dataset.data.x.shape[-1])

        # # Remove the outliers from dataset.data.x and dataset.data.y
        # # dataset.data.x = torch.tensor([x for i, x in enumerate(dataset.data.x) if i not in indices_to_remove])
        # dataset.data.x = torch.vstack([x for i, x in enumerate(dataset.data.x) if i not in indices_to_remove])
        # dataset.data.y = torch.tensor([y for i, y in enumerate(dataset.data.y) if i not in indices_to_remove])

        # # Recalculate min and max values after removing outliers
        # min_value = torch.min(dataset.data.y)
        # max_value = torch.max(dataset.data.y)
        #---------------------------------

        # Normalize the remaining data to the range of 0 to 1
        # dataset.data.y = (dataset.data.y - min_value) / (max_value - min_value)



        # Calculate min and max values
        # min_value = torch.min(dataset.data.y)
        # max_value = torch.max(dataset.data.y)

        # #Normalize the data to the range of 0 to 1
        # dataset.data.y = (dataset.data.y - min_value) / (max_value - min_value)

        isThereNaN = torch.isnan(dataset.data.x).any().item()
        print("Does NaN Exist? :" + str(isThereNaN))

        
        # num_bins = 20
        # num_nodes = 100
        # num_time_points = dataset.data.x.shape[-1]
        # num_subs = dataset.data.x.shape[0]/num_nodes

        # # Initialize the SAX transformer
        # sax = SymbolicAggregateApproximation(n_bins=num_bins, strategy='uniform')

        # # Reshape the data to fit the transformer
        # reshaped_data = dataset.data.x

        # # Apply SAX transformation
        # sax_data = sax.transform(reshaped_data.detach().cpu().numpy())

        # sax_numeric = np.array([list(map(ord, seq)) for seq in sax_data])



        tr_index,val_index,te_index = train_val_test_split(fold=0, n_subjects = dataset.data.y.shape[0] )
        train_dataset = dataset[tr_index]
        val_dataset = dataset[val_index]
        test_dataset = dataset[te_index]

        self.train = ABCDDGL(train_dataset)
        self.val = ABCDDGL(val_dataset)
        self.test = ABCDDGL(test_dataset)
        
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    #     if os.path.exists(save_path):
    #         print("Loading dataset %s from %s..." % (name, save_path))
    #         with open(save_path, "rb") as f:
    #             saved_datasets = pickle.load(f)
    #             self.train = saved_datasets['train']
    #             self.val = saved_datasets['val']
    #             self.test = saved_datasets['test']
    #     else:
    #         self._load_and_process_data()
    #         #self._save_datasets()

    # def _load_and_process_data(self):
    #     # Load and preprocess your dataset
    #     # ... (your dataset loading and processing code) ...
    #     start = time.time()
    #     data_path = "/data/users2/bthapaliya/PEIntelligenceABCDPredictionBrainGNN/data/data/Output"
    #     dataset = ABIDEDatasetPECryst(data_path, "ABCD")

    #     tr_index,val_index,te_index = train_val_test_split(fold=0, n_subjects = dataset.data.y.shape[0] )
    #     train_dataset = dataset[tr_index]
    #     val_dataset = dataset[val_index]
    #     test_dataset = dataset[te_index]

    #     self.train = ABCDDGL(train_dataset)
    #     self.val = ABCDDGL(val_dataset)
    #     self.test = ABCDDGL(test_dataset)
        
    #     print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
    #     print("[I] Finished loading.")
    #     print("[I] Data load time: {:.4f}s".format(time.time() - start))


    # def _save_datasets(self):
    #     saved_datasets = {'train': self.train, 'val': self.val, 'test': self.test}
    #     with open(self.save_path, "wb") as f:
    #         pickle.dump(saved_datasets, f)
    #     print("Datasets saved to %s" % self.save_path)

   
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    

    def _laplace_decomp(self, max_freqs):
        self.train.graph_lists = [laplace_decomp(g, max_freqs) for g in self.train.graph_lists]
        self.val.graph_lists = [laplace_decomp(g, max_freqs) for g in self.val.graph_lists]
        self.test.graph_lists = [laplace_decomp(g, max_freqs) for g in self.test.graph_lists]
    

    def _make_full_graph(self):
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]


    def _add_edge_laplace_feats(self):
        self.train.graph_lists = [add_edge_laplace_feats(g) for g in self.train.graph_lists]
        self.val.graph_lists = [add_edge_laplace_feats(g) for g in self.val.graph_lists]
        self.test.graph_lists = [add_edge_laplace_feats(g) for g in self.test.graph_lists]        

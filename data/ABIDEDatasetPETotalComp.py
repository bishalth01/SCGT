import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from data.read_abide_stats_parall_pe import read_data


class ABIDEDatasetPETotalComp(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDatasetPETotalComp, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #data_dir = osp.join(self.root,'53_raw_fluid_sitecorrected')
        #data_dir = osp.join(self.root,'100_nodes_timeseries_intelligence_totalcomp_notcorrected')
        data_dir = "/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN/data/data/Output/100_raw_totalcomp_sitecorrected"
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        #return  '53_data_cryst_sitecorrected.pt'
        #return  '100_nodes_timeseries_intelligence_cryst_sanpositional_eigenvecs.pt'

        return  '100_nodes_timeseries_intelligence_totalcomp_sanpositional_eigenvecs_nooutliers_final_notcorrected.pt'    
    
    # def pre_filter(self, data):
    #     # Filter out data samples where label is not greater than 58
    #     return data.y > 58

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.raw_dir
        #self.data, self.slices = read_data("/data/users2/bthapaliya/PEIntelligenceABCDPredictionBrainGNN/data/data/Output/100_nodes_timeseries_intelligence_totalcomp_notcorrected")
        self.data, self.slices = read_data("/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN/data/data/Output/100_raw_totalcomp_sitecorrected")
        self.pre_filter= True

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            # data_list = [data for data in data_list if (data.y>58)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

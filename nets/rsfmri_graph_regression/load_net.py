

import sys

sys.path.append("/data/users2/bthapaliya/SAN-main/SAN-main")
#from nets.rsfmri_graph_regression.SAN_NodeNoLPE  import SAN_NodeNoLPE
from nets.rsfmri_graph_regression.SAN_NodeLPEWithPE import SAN_NodeLPEWithPosEnc
from nets.rsfmri_graph_regression.SAN_NodeLPEWithPECommunity import SAN_NodeLPEWithPosEncCommunity
from nets.rsfmri_graph_regression.SAN_NodeLPE import SAN_NodeLPE
# from nets.rsfmri_graph_regression.SAN_NodeNoLPE  import SAN_NodeNoLPE
from nets.rsfmri_graph_regression.SAN_EdgeLPE import SAN_EdgeLPE
from nets.rsfmri_graph_regression.SAN import SAN
from nets.rsfmri_graph_regression.SANGRU import SANGRU
from nets.rsfmri_graph_regression.SAN_NodeLPEWithPosEncGRU import SAN_NodeLPEWithPosEncGRU
from nets.rsfmri_graph_regression.SAN_NodeLPEWithPECommunityClassification import SAN_NodeLPEWithPosEncCommunityClassification
from nets.rsfmri_graph_regression.SAN_NodeLPEWithPECommunityTopK import SAN_NodeLPEWithPosEncCommunityTopK

def CommunityLPE(net_params):
    return SAN_NodeLPEWithPosEncCommunity(net_params)

def CommunityLPETopK(net_params):
    return SAN_NodeLPEWithPosEncCommunityTopK(net_params)


def CommunityLPEClassification(net_params):
    return SAN_NodeLPEWithPosEncCommunityClassification(net_params)

def NodeLPE(net_params):
    return SAN_NodeLPEWithPosEnc(net_params)

def NodeGRU(net_params):
    return SAN_NodeLPEWithPosEncGRU(net_params)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return SAN_NodeLPE(net_params)

def gnn_model(LPE, net_params):
    model = {
        'edge': EdgeLPE,
        'node': NodeLPE,
        'none': NoLPE,
        'gru': NodeGRU,
        'community': CommunityLPE,
        'sex': CommunityLPEClassification,
        'communitytopk': CommunityLPETopK
    }
        
    return model[LPE](net_params)
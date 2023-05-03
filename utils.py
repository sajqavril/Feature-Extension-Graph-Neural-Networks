import os
import torch
import numpy as np
from torch_geometric.utils.loop import add_self_loops
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, Actor, Coauthor, Amazon, PPI, Reddit2, Yelp, AmazonProducts
from ogb.nodeproppred import PygNodePropPredDataset


def get_data(args):

    name = args.dataset
    split = args.split
    train_proportion = args.train_proportion
    val_proportion = args.val_proportion
    idx = args.idx

    name = name.lower()
    path = './dataset'
    if name == 'actor':
        path = os.path.join(path, name)
    transforms = T.Compose([]) # T.NormalizeFeatures()
    dataset_func = {
        'cora': Planetoid,
        'citeseer': Planetoid,
        'pubmed': Planetoid,
        'actor': Actor,
        'computers': Amazon, 
        'photo': Amazon,     
    }
    data_npz = {
        'chameleon': './dataset/chameleon.npz',
        'squirrel': './dataset/squirrel.npz',
    }
    
    if name not in ["chameleon", "squirrel", "actor"]:
        dataset = dataset_func[name](path, name=name, transform=transforms)
        data = dataset[idx]
    elif name in ["actor"]:
        dataset = dataset_func[name](path, transform=transforms)
        data = dataset[idx]
    else:
        data = load_npz_data(data_npz[name])
        row_norm = data.x.sum(dim=1, keepdim=True).clamp_(min=1.)
        data.x.div_(row_norm)
        
    if (name in ['cora', 'pubmed', 'citeseer']) and (split == 'grand'):
        pass

    else: # for now, one split is used
        num_class = data.y.max() + 1
        n = data.x.shape[0]
        val_lb = int(n * val_proportion)
        percls_trn = int(train_proportion * n / num_class)
        data = random_planetoid_splits(data, num_class, percls_trn, val_lb, seed=args.seed)


    # # get compressed adj
    edges, weight = get_laplacian(edge_index=data.edge_index, normalization='sym', num_nodes=data.x.shape[0])
    edges, weight = add_self_loops(edge_index=edges, fill_value=1., edge_weight=-weight, num_nodes=data.x.shape[0])
    adj = torch.sparse_coo_tensor(indices=data.edge_index, 
                                    values=torch.ones_like(data.edge_index[0]),
                                    size=(data.x.shape[0], data.x.shape[0]), 
                                    device=data.x.device,
                                    dtype=torch.float).to_dense()
    adj = F.normalize(adj, dim=1)
    U, S, V = torch.svd_lowrank(adj, q=args.nl)
    adj = torch.mm(U, torch.diag(S))
    
    adj = F.normalize(adj, dim=0)
    data.adj = adj


    return data


def load_npz_data(path):
    raw = dict(np.load(path, allow_pickle=True))
    data = Data(x=torch.Tensor(raw['features']), y=torch.LongTensor(raw['label']), edge_index=torch.LongTensor(raw['edges']).t())
    del raw

    return data

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data


def set_best_train_args(args):
    table = pd.read_csv('FEGNN_params.csv', delimiter=',', engine='python')
    name = args.dataset.lower()
    
    ind = (table['dataset']==name).to_numpy().nonzero()[0]
    args.lr = float(table['lr'][ind])
    args.weight_decay = float(table['weight_decay'][ind])
    args.nhid = int(table['nhid'][ind])
    args.nl = int(table['S'][ind])
    args.K = int(table['layers'][ind])
    return args


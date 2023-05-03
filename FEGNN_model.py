import torch
from torch import svd_lowrank
import torch.nn as nn
import torch.nn.functional as F
import math

from scipy.special import comb
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, add_self_loops

class MP(MessagePassing):
    def __init__(self):
        super(MP, self).__init__()

    def message(self, x_j, norm=None):
        if norm != None:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp.data.tolist()) 



class BasisGenerator(nn.Module):
    '''
    generate all the feature spaces
    '''
    def __init__(self, nx, nlx, nl, K, poly, low_x=False, low_lx=False, low_l=True, norm1=False):
        super(BasisGenerator, self).__init__()
        self.nx = nx
        self.nlx = nlx
        self.nl = nl
        self.norm1 = norm1
        self.K = K # for lx
        self.poly = poly # for lx
        self.low_x = low_x
        self.low_lx = low_lx
        self.low_l = low_l
        self.mp = MP()

    def get_x_basis(self, x):
        x = F.normalize(x, dim=1)
        x = F.normalize(x, dim=0)
        if self.low_x:
            U, S, V = svd_lowrank(x, q=self.nx)
            low_x = torch.mm(U, torch.diag(S))
            return low_x
        else:
            return x

    def get_lx_basis(self, x, edge_index):
        # generate all feature spaces
        lxs = []
        num_nodes = x.shape[0]
        lap_edges, lap_norm = get_laplacian(edge_index=edge_index,
                                                normalization='sym',
                                                num_nodes=num_nodes) # 标准的归一化后lap
        h = F.normalize(x, dim=1)

        if self.poly == 'gcn':
            lxs = [h]
            # 
            edges, norm = add_self_loops(edge_index=lap_edges,
                                            edge_weight=-lap_norm,
                                            fill_value=2.,
                                            num_nodes=num_nodes) # \hat{A} = I + \tilde{A}
            edges, norm = get_laplacian(edge_index=edges,
                                            edge_weight=norm,
                                            normalization='sym',
                                            num_nodes=num_nodes) # \hat{L}
            edges, norm = add_self_loops(edge_index=edges,
                                            edge_weight=-norm,
                                            fill_value=1.,
                                            num_nodes=num_nodes)
            # may use directly gcn-norm
            # gcn_norm(edge_index=edge_index, num_nodes=num_nodes) 

            for k in range(self.K + 1):
                h = self.mp.propagate(edge_index=edges, x=h, norm=norm) 
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'gpr':
            lxs = [h]
            edges, norm = add_self_loops(edge_index=lap_edges,
                                            edge_weight=-lap_norm,
                                            fill_value=1.,
                                            num_nodes=num_nodes)
            for k in range(self.K):
                h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'ours':
            lxs = [h]
            edges, norm = add_self_loops(edge_index=lap_edges,
                                            edge_weight=lap_norm,
                                            fill_value=-1.,
                                            num_nodes=num_nodes)
            for k in range(self.K):
                h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)
        
        elif self.poly == 'cheb':
            edges, norm = add_self_loops(edge_index=lap_edges,
                                            edge_weight=lap_norm,
                                            fill_value=-1.,
                                            num_nodes=num_nodes)
            for k in range(self.K + 1):
                if k == 0:
                    pass
                elif k == 1:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                else:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm) * 2
                    h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'cheb2':
            # 
            tlx = [h] 
            edges, norm = add_self_loops(edge_index=lap_edges,
                                            edge_weight=lap_norm,
                                            fill_value=-1.,
                                            num_nodes=num_nodes)
            for k in range(self.K):
                if k == 0:
                    pass
                elif k == 1:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                    
                else:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm) * 2
                    h = h - tlx[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                tlx.append(h)
            # 
            for j in range(self.K + 1):
                lxs.append(0)
                # 
                xjs = [] 
                xj = math.cos((j + 0.5) * torch.pi / (self.K + 1))
                for i in range(self.K + 1):
                    if i == 0:
                        xjs.append(1)
                    elif i == 1:
                        xjs.append(xj)
                    else:
                        tmp = 2 * xj * xjs[-1] - xjs[-2]
                        xjs.append(tmp)
                    lxs[-1] = lxs[-1] + tlx[i] * xjs[-1]
                
        elif self.poly == 'bern':
            edges1, norm1 = lap_edges, lap_norm
            edges2, norm2 = add_self_loops(edge_index=lap_edges,
                                            edge_weight=-lap_norm,
                                            fill_value=2.,
                                            num_nodes=num_nodes)
            tmps = [h]
            for k in range(self.K):
                h = self.mp.propagate(edge_index=edges1, x=h, norm=norm1) 
                tmps.append(h)
            # all feature spaces
            for i, h in enumerate(tmps):
                tmp = h
                for j in range(self.K - i):
                    tmp = self.mp.propagate(edge_index=edges2, x=tmp, norm=norm2)
                tmp = tmp * comb(self.K, i) / 2 ** self.K
                lxs.append(tmp)


        # 
        normed_lxs = []
        low_lxs = []
        for lx in lxs:
            if self.low_lx:
                U, S, V = svd_lowrank(lx)
                low_lx = torch.mm(U, torch.diag(S))
                low_lxs.append(low_lx)
                normed_lxs.append(F.normalize(low_lx, dim=1))
            else:
                normed_lxs.append(F.normalize(lx, dim=1))
        
        # final_lx = [F.normalize(lx, dim=1) for lx in normed_lxs] # norm1
        final_lx = [F.normalize(lx, dim=0) for lx in lxs] # no norm1
        return final_lx


    def get_l_basis(self, edge_index, num_nodes, adj):
        if self.low_l:
            return adj
        # use adj
        l = torch.sparse_coo_tensor(indices=edge_index,
                                            values=torch.ones_like(edge_index[0]),
                                            size=(num_nodes, num_nodes),
                                            device=edge_index.device)
        # use lap
        lap_edges, lap_norm = get_laplacian(edge_index=edge_index,
                                                normalization='sym',
                                                num_nodes=num_nodes)
        l = torch.sparse_coo_tensor(indices=lap_edges,
                                            values=lap_norm,
                                            size=(num_nodes, num_nodes),
                                            device=edge_index.device).to_dense()
        if self.low_l:
            l = F.normalize(l, dim=1)
            U, S, V = svd_lowrank(l, q=self.nl)
            low_l = torch.mm(U, torch.diag(S))
            low_l = F.normalize(low_l, dim=0)
            return low_l
        else:
            l = F.normalize(l, dim=0)
            return l


class FEGNN(nn.Module):

    def __init__(self, ninput, nclass, args):
        super(FEGNN, self).__init__()
        self.K = args.K
        self.poly = args.poly
        self.nx = ninput if args.nx < 0 else args.nx # nx >= 0
        self.nlx = ninput if args.nlx < 0 else args.nlx # nlx >= 0
        self.nl = args.nl # nl 
        self.lin_x = nn.Linear(self.nx, args.nhid, bias=True)
        self.lin_lx = nn.Linear(self.nlx, args.nhid, bias=True)
        self.lin_l = nn.Linear(self.nl, args.nhid, bias=True)
        self.lin2 = nn.Linear(args.nhid, nclass, bias=True)
        self.basis_generator = BasisGenerator(nx=self.nx, nlx=self.nlx, nl=self.nl, K=args.K, poly=args.poly,
                                                low_x=False, low_lx=False, low_l=True, norm1=False)
        self.thetas = nn.Parameter(torch.ones(args.K + 1), requires_grad=True)
        self.lin_lxs = torch.nn.ModuleList()
        for i in range(self.K + 1):
            self.lin_lxs.append(nn.Linear(self.nlx, args.nhid, bias=True))
        self.share_lx = args.share_lx

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, edge_index, cp_adj = data.x, data.edge_index, data.adj
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)[0:] # []
        l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0], cp_adj)

        dict_mat = 0

        if self.nx > 0:
            x_dict = self.lin_x(x_basis)
            dict_mat = dict_mat + x_dict

        if self.nlx > 0:
            lx_dict = 0
            for k in range(self.K + 1):
                if self.share_lx:
                    lx_b = self.lin_lx(lx_basis[k]) * self.thetas[k] # share W_lx across each layer/order
                else:
                    lx_b = self.lin_lxs[k](lx_basis[k]) # do not share the W_lx parameters
                lx_dict = lx_dict + lx_b
            dict_mat = dict_mat + lx_dict

        if self.nl > 0:
            l_dict = self.lin_l(l_basis)
            dict_mat = dict_mat + l_dict


        res = self.lin2(dict_mat)

        return F.log_softmax(res, dim=1)


    def get_dict(self, data):
        x, edge_index, cp_adj = data.x, data.edge_index, data.adj
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)[0:] # []
        l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0], cp_adj)

        dict_mat = 0
        dict0 = []
        
        if self.nlx > 0:
            for k in range(self.K + 1):
                lx_dict = 0
                if self.share_lx:
                    lx_b = self.lin_lx(lx_basis[k]) * self.thetas[k] # share W_lx across each layer/order
                else:
                    lx_b = self.lin_lxs[k](lx_basis[k]) # do not share the W_lx parameters
                lx_dict = lx_dict + lx_b
                dict0.append(lx_basis[k])
            dict_mat = dict_mat + lx_dict
            

        if self.nx > 0:
            x_dict = self.lin_x(x_basis)
            dict_mat = dict_mat + x_dict
            dict0.append(x_basis)

        if self.nl > 0:
            l_dict = self.lin_l(l_basis)
            dict_mat = dict_mat + l_dict
            dict0.append(l_basis)

        dict0 = torch.cat(dict0, dim=1)

        return dict0, dict_mat

 
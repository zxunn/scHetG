import torch
import torch.nn as nn

import dgl.function as fn
import torch.nn.functional as F


class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """
        self.act = nn.Sigmoid()

    def forward(self, graph, ufeats, ifeats, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        ufeats : torch.FloatTensor
            cell features
        ifeats : torch.FloatTensor
            gene features

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_cells, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = ufeats
            graph.nodes[gkey].data['h'] = ifeats
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pred = self.act(graph.edata['score'])

        return pred

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x)-1., min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class ZINBDecoder(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """ZINB decoder for link prediction
        predict link existence (not edge type)
        """
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_disp = nn.Linear(feats_dim, 1)
        self.dec_disp_act = DispAct()
        self.dec_pi = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()

    def forward(self, graph, ufeats, ifeats, ckey, gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        ufeats : torch.FloatTensor
            cell features
        ifeats : torch.FloatTensor
            gene features

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_cells, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = ufeats
            graph.nodes[gkey].data['h'] = ifeats
            graph.nodes[ckey].data['one'] = torch.ones([ufeats.shape[0], 1], device=ufeats.device)
            graph.nodes[gkey].data['one'] = torch.ones([ifeats.shape[0], 1], device=ifeats.device)

            graph.apply_edges(fn.u_mul_v('h', 'h', 'h_d'))

            h_d = graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            disp_ = self.dec_disp(h_d)
            pi = self.dec_pi(h_d)

            graph.apply_edges(fn.u_mul_v('one', 'ge_factor', 'ge_factor'))
            graph.apply_edges(fn.u_mul_v('sz_factor', 'one', 'sz_factor'))
            disp = self.dec_disp_act(graph.edata['ge_factor'] * disp_)
            mu_ = graph.edata['ge_factor'] * mu_
            mu = graph.edata['sz_factor'] * self.dec_mean_act(mu_)

        return mu, disp, pi

class MLP_Decoder(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """ZINB decoder for link prediction
        """
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_disp = nn.Linear(feats_dim, 1)
        self.dec_disp_act = DispAct()
        self.dec_pi = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()

    def forward(self, graph, ufeats, n_genes, ckey, gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        ufeats : torch.FloatTensor
            cell features
        ifeats : torch.FloatTensor
            gene features

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_cells, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = ufeats
            graph.nodes[ckey].data['one'] = torch.ones([ufeats.shape[0], 1], device=ufeats.device)
            graph.nodes[gkey].data['one'] = torch.ones([n_genes, 1], device=ufeats.device)

            graph.apply_edges(fn.u_mul_v('h', 'one', 'h_d'))

            h_d = graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            disp_ = self.dec_disp(h_d)
            pi = self.dec_pi(h_d)

            disp = self.dec_disp_act(disp_)
            mu = self.dec_mean_act(mu_)

        return mu, disp, pi


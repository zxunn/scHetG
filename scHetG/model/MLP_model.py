import torch
from torch import nn
from .decoder import DotDecoder, MLP_Decoder


class MLP(nn.Module):
    def __init__(self,
                 feats_dim=64,
                 drop_out=0.1):
        """Light Graph Convolution

        Paramters
        ---------
        drop_out : float
            dropout rate (neighborhood dropout)
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(feats_dim, feats_dim),
            nn.Dropout(drop_out),
            nn.BatchNorm1d(feats_dim),
            nn.ReLU()
        ).to(torch.device('cuda'))


    def forward(self, feats):
        """Apply Light Graph Convoluiton to specific edge type {r}

        Paramters
        ---------
        graph : dgl.graph
        src_feats : torch.FloatTensor
            source node features

        ci : torch.LongTensor
            in-degree of sources ** (-1/2)
            shape : (n_sources, 1)
        cj : torch.LongTensor
            out-degree of destinations ** (-1/2)
            shape : (n_destinations, 1)

        Returns
        -------
        output : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
                where N_{i, r} ; number of neighbors_{i, r} ** (1/2)
        2. aggregation
            \sum_{j \in N(i), r} MP_{j -> i, r}
        """
        out = self.linear(feats)

        return out


class MLPLayer(nn.Module):
    def __init__(self,
                 ckey,
                 feats_dim=64,
                 drop_out=0.1):
        super().__init__()
        """LightGCN Layer

        edge_types : list
            all edge types
        drop_out : float
            dropout rate (feature dropout)
        """
        self.ckey = ckey
        self.n_batch = len(ckey)
        conv = {}

        for i in range(len(ckey)):

            cell_key = 'cell' + str(i + 1)
            conv[cell_key] = MLP(feats_dim=feats_dim, drop_out=drop_out)

        self.conv = conv
        self.feature_dropout = nn.Dropout(drop_out)
        self.normed_adj_u, self.normed_adj_v = None, None

    def forward(self, ufeats, ckey=['cell1', 'cell2']):
        """
        Paramters
        ---------
        graph : dgl.graph
        ufeats, ifeats : torch.FloatTensor
            node features
        ckey, gkey : str
            target node types

        Returns
        -------
        ufeats, ifeats : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma_{j \in N(i) , r} MP_{i, j, r}
        """
        feats = {
            ckey[i]: ufeats[i]
            for i in range(self.n_batch)
        }

        out = [self.conv[key](feats[key]) for key in ckey]

        return out


class MLP_model(nn.Module):
    def __init__(self,
                 n_layers,
                 n_cells,
                 n_genes,
                 drop_out,
                 feats_dim,
                 decoder='ZINB',
                 learnable_weight=False):
        super().__init__()
        """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
        paper : https://arxiv.org/pdf/2002.02126.pdf

        n_layers : int
            number of GCMC layers
        edge_types : list
            all edge types
        drop_out : float
            dropout rate (neighbors)
        learnable_weight : boolean
            whether to learn weights for embedding aggregation
            if False, use 1/n_layers
        """

        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_batch = len(n_cells)
        self.ckey = ['cell{}'.format(i + 1) for i in range(self.n_batch)]
        self.gkey = 'gene'

        self.cell_feature = nn.ParameterList([nn.Parameter() for _ in range(self.n_batch)])
        self.batch = [None] * self.n_batch
        for i in range(len(n_cells)):
            self.cell_feature[i] = nn.Parameter(torch.Tensor(self.n_cells[i], feats_dim))
            batch_ = torch.zeros(self.n_cells[i], len(n_cells))
            batch_[:, i] = torch.ones(self.n_cells[i])
            self.batch[i] = batch_.to(torch.device('cuda'))

        self.emb_layer = nn.Sequential(nn.Linear(feats_dim + self.n_batch, feats_dim), nn.BatchNorm1d(feats_dim),
                                       nn.ELU())

        self.pred_pos = [None] * self.n_batch
        self.pred_neg = [None] * self.n_batch
        self.u_hidden = [None] * self.n_batch
        self.h_cell = [None] * self.n_batch

        for i in range(self.n_batch):
            nn.init.xavier_uniform_(self.cell_feature[i])

        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(MLPLayer(ckey=self.ckey, feats_dim=feats_dim, drop_out=drop_out))


        if self.n_layers == 2:
            self.weights = torch.tensor([1. / 3, 1. / 3, 1. / 3])
        else:
            self.weights = torch.tensor([1., 1. / 2, 1., 1. / 2, 1.])

        if learnable_weight:
            self.weights = nn.Parameter(self.weights)

        if decoder == 'Dot':
            self.decoder = DotDecoder()
        elif decoder == 'ZINB':
            self.decoder = MLP_Decoder(feats_dim=feats_dim)

        for p, q in self.decoder.named_parameters():
            if 'weight' in p:
                nn.init.kaiming_normal_(q)
            elif 'bias' in p:
                nn.init.constant_(q, 0)


    def encode(self, ufeats, ckey):

        for i in range(self.n_batch):
            self.u_hidden[i] = self.weights[0] * ufeats[i]

        for i in range(self.n_batch):
            self.h_cell[i] = self.weights[0] * ufeats[i]

        for w, encoder in zip(self.weights[1:], self.encoders):
            ufeats = encoder(ufeats, ckey)

            for i in range(self.n_batch):
                self.u_hidden[i] = self.u_hidden[i] + w * ufeats[i]


        for i in range(self.n_batch):
            self.h_cell[i] = self.h_cell[i] + self.weights[2] * ufeats[i]

        return self.u_hidden, self.h_cell

    def decode(self, pos_graph, neg_graph, ufeats, n_genes, ckey, gkey):

        for i in range(self.n_batch):
            self.pred_pos[i] = self.decoder(pos_graph[i], ufeats[i], n_genes, ckey[i], gkey)
            self.pred_neg[i] = self.decoder(neg_graph[i], ufeats[i], n_genes, ckey[i], gkey)

        return self.pred_pos, self.pred_neg

    def forward(self,
                pos_graph,
                neg_graph=None
                ):
        """
        Parameters
        ----------
        enc_graph : dgl.graph
        dec_graph : dgl.homograph

        Notes
        -----
        1. LightGCN encoder
            1 ) message passing
                MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
            2 ) aggregation
                \sum_{j \in N(i), r} MP_{j -> i, r}

        2. final features
            cell_{i} = mean( h_{i, layerself.cell_feature = {Parameter: (943, 75)} Parameter containing:\ntensor([[ 0.0007, -0.0501,  0.0644,  ..., -0.0756,  0.0526, -0.0293],\n        [ 0.0743, -0.0693, -0.0382,  ..., -0.0612,  0.0300,  0.0068],\n        [-0.0341, -0.0038,  0.0670,  ..., -0.0470, -0.0631, -0.0403],\n        ...,\n        [-0… View_1}, h_{i, layer_2}, ... )
            gene_{j} = mean( h_{j, layer_1}, h_{j, layer_2}, ... )

        3. Bilinear decoder
            logits_{i, j, r} = ufeats_{i} @ Q_r @ ifeats_{j}
        """


        ufeats, h_cell = self.encode(self.cell_feature, self.ckey)

        self.emb_cell = ufeats

        ufeats = [self.emb_layer(torch.cat((ufeats[i], self.batch[i]), dim=1)) for i in range(self.n_batch)]

        if neg_graph:
            pred_pos, pred_neg = self.decode(pos_graph, neg_graph, ufeats, self.n_genes, self.ckey, self.gkey)
            return pred_pos, pred_neg
        else:
            for i in range(self.n_batch):
                self.pred_pos[i] = self.decoder(pos_graph[i], ufeats[i], self.n_genes, self.ckey[i], self.gkey)

        return self.pred_pos
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv, GINConv, GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
{
    'in_dim'
    'hid_dim'
    'n_classes'
    'activation'
    'node_each_graph'
    'n_layers'
    'dropout'
    
    'learn_eps'
    'n_mlp_layers'
    'neighbor_pooling_type'
    'graph_pooling_type'
    'final_dropout'
    
    'heads'
    'feat_drop'
    'attn_drop'
    'negative_slope'
    'residual'
}
"""

# GCN Model
class GCN(nn.Module):
    def __init__(self, model_config):
        super(GCN, self).__init__()

        in_feats = model_config['in_dim']
        n_hidden = model_config['hid_dim']
        n_classes = model_config['n_classes']
        n_layers = model_config['n_layers']
        node_each_graph = model_config['node_each_graph']
        activation = model_config['activation']
        dropout = model_config['dropout']

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        # 投影层
        self.classification = nn.Linear(node_each_graph * n_hidden, n_classes)
        self.node_each_graph = node_each_graph
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, batch_size):
        h = graph.ndata['feat'].to(torch.float32)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = h.reshape(batch_size, -1)
        logits = self.classification(h)
        return logits


    def get_embedding(self, graph, batch_size):
        h = graph.ndata['feat'].to(torch.float32)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = h.reshape(batch_size, -1)
        return h


# GIN Model
class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    def __init__(self, model_config):
        super(GIN, self).__init__()
        self.n_layers = model_config['n_layers']
        self.learn_eps = model_config['learn_eps']
        self.node_each_graph = model_config['node_each_graph']

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.n_layers - 1):
            if layer == 0:
                mlp = MLP(model_config['n_mlp_layers'], model_config['in_dim'], model_config['hid_dim'], model_config['hid_dim'])
            else:
                mlp = MLP(model_config['n_mlp_layers'], model_config['hid_dim'], model_config['hid_dim'], model_config['hid_dim'])

            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), model_config['neighbor_pooling_type'], 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(model_config['hid_dim']))
        self.linears_prediction = torch.nn.ModuleList()
        self.classification = nn.Linear(self.node_each_graph * model_config['hid_dim'], model_config['n_classes'])

        for layer in range(self.n_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(model_config['node_each_graph'] * model_config['node_each_graph'], model_config['n_classes'])
                )
            else:
                self.linears_prediction.append(
                    nn.Linear(model_config['node_each_graph'] * model_config['hid_dim'], model_config['n_classes'])
                )

        self.drop = nn.Dropout(model_config['final_dropout'])
        if model_config['graph_pooling_type'] == 'sum':
            self.pool = SumPooling()
        elif model_config['graph_pooling_type'] == 'mean':
            self.pool = AvgPooling()
        elif model_config['graph_pooling_type'] == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, graph, batch_size):
        h = graph.ndata['feat'].to(torch.float32)
        hidden_rep = [h]

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](graph, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = h.reshape(batch_size, -1)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer


# GAT
class GAT(nn.Module):
    def __init__(self, model_config):
        super(GAT, self).__init__()
        self.num_layers = model_config['n_layers']
        self.gat_layers = nn.ModuleList()
        self.activation = model_config['activation']
        self.node_each_graph = model_config['node_each_graph']
        heads = model_config['heads']
        # num_layers = model_config['num_layers']

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            model_config['in_dim'], model_config['hid_dim'], heads[0],
            model_config['feat_drop'], model_config['attn_drop'], model_config['negative_slope'], False,
            self.activation, bias=False, allow_zero_in_degree=False))
        # hidden layers
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                model_config['hid_dim'] * heads[l-1], model_config['hid_dim'], heads[l],
                model_config['feat_drop'], model_config['attn_drop'], model_config['negative_slope'],
                model_config['residual'], self.activation, bias=False, allow_zero_in_degree=False))
        self.classification = nn.Linear(self.node_each_graph * model_config['hid_dim'], model_config['n_classes'])


    def forward(self, graph, batch_size):
        h = graph.ndata['feat']
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](graph, h).flatten(1)
        h = self.gat_layers[-1](graph, h).mean(1)
        # output projection
        h = h.reshape(batch_size, -1)
        logits = self.classification(h)
        return logits


# VGAE
class VGAE(nn.Module):
    def __init__(self, model_config):
        super(VGAE, self).__init__()
        self.in_dim = model_config['in_dim']
        self.hid_dim = model_config['hid_dim']
        self.node_each_graph = model_config['node_each_graph']
        layers = [GraphConv(self.in_dim, self.hid_dim, activation=F.relu, norm='none', allow_zero_in_degree=True),
                  GraphConv(self.hid_dim, self.hid_dim, activation=lambda x: x, norm='none', allow_zero_in_degree=True),
                  GraphConv(self.hid_dim, self.hid_dim, activation=lambda x: x, norm='none', allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        self.classification = nn.Linear(self.node_each_graph * model_config['hid_dim'], model_config['n_classes'])

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hid_dim)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def encode_decode(self, graph):
        feat = graph.ndata['feat'].to(torch.float32)
        z = self.encoder(graph, feat)
        adj_rec = self.decoder(z)
        return adj_rec

    def forward(self, graph, batch_size):
        feat = graph.ndata['feat']
        h = self.layers[0](graph, feat)
        h = h.reshape(batch_size, -1)
        logits = self.classification(h)
        return logits

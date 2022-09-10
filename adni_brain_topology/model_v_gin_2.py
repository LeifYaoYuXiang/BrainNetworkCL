import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


# 使用GIN 作为我们的Encoder
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


class Model_v2(nn.Module):
    def __init__(self, model_config):
        super(Model_v2, self).__init__()
        self.n_layers = model_config['n_layers']
        self.learn_eps = model_config['learn_eps']
        self.node_each_graph = model_config['node_each_graph']

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()
        self.fc_layer = nn.Linear(model_config['n_hidden'], model_config['n_classes'])

        for layer in range(self.n_layers - 1):
            if layer == 0:
                mlp = MLP(model_config['n_mlp_layers'], model_config['in_feats'], model_config['n_hidden'],
                          model_config['n_hidden'])
            else:
                mlp = MLP(model_config['n_mlp_layers'], model_config['n_hidden'], model_config['n_hidden'],
                          model_config['n_hidden'])

            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), model_config['neighbor_pooling_type'], 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(model_config['n_hidden']))

        # CHANGE: model_config['n_class'] -> model_config['n_hidden']
        for layer in range(self.n_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(model_config['node_each_graph'] * model_config['node_each_graph'],
                              model_config['n_hidden'])
                )
            else:
                self.linears_prediction.append(
                    nn.Linear(model_config['node_each_graph'] * model_config['n_hidden'], model_config['n_hidden'])
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

    def forward(self, batch_graph, batch_size):
        h = batch_graph.ndata['feat'].to(torch.float32)
        hidden_rep = [h]

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](batch_graph, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = h.reshape(batch_size, -1)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        score_over_layer = self.fc_layer(score_over_layer)
        return score_over_layer

    # get embedding
    def get_embedding(self, batch_graph, batch_size):
        h = batch_graph.ndata['feat'].to(torch.float32)
        hidden_rep = [h]

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](batch_graph, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = h.reshape(batch_size, -1)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer


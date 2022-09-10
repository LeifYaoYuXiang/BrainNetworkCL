import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.conv import GraphConv, GINConv, GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


# GCN Model
class Model_v3(nn.Module):
    def __init__(self, model_config):
        super(Model_v3, self).__init__()

        in_feats = model_config['in_feats']
        n_hidden = model_config['n_hidden']
        n_classes = model_config['n_classes']
        n_layers = model_config['n_layers']
        node_each_graph = model_config['node_each_graph']
        activation = F.relu
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
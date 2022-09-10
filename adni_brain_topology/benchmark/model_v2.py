import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl.function as fn


# reference: https://github.com/raspberryice/ala-gcn/blob/8fc8a747a05740983c8af2ab78a471df61a72f91/layers.py
# reference: https://github.com/raspberryice/ala-gcn/blob/8fc8a747a05740983c8af2ab78a471df61a72f91/network.py
def adaptive_reduce_func(nodes):
    _, pred = torch.max(nodes.mailbox['logits'], dim=2)
    _, center_pred = torch.max(nodes.data['logits'], dim=1)
    # n_degree = nodes.data['degree']
    n_degree = 1
    # case 1
    # ratio of common predictions
    f1 = torch.sum(torch.eq(pred,center_pred.unsqueeze(1)), dim=1)/n_degree
    f1 = f1.detach()
    # case 2
    # entropy of neighborhood predictions
    uniq = torch.unique(pred)
    cnts_p = torch.zeros((pred.size(0), uniq.size(0),))
    for i,val in enumerate(uniq):
        tmp = torch.sum(torch.eq(pred, val), dim=1)/n_degree
        cnts_p[:, i] = tmp
    cnts_p = torch.clamp(cnts_p, min=1e-5)

    f2 = (-1)* torch.sum(cnts_p * torch.log(cnts_p),dim=1)
    f2 = f2.detach()
    return {
        'f1': f1,
        'f2':f2,
    }


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        # h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        # h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GatedLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, num_nodes, lidx=1):
        super(GatedLayer, self).__init__()
        self.weight_neighbors = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.tau_1 = nn.Parameter(torch.zeros((1,)))
        self.tau_2 = nn.Parameter(torch.zeros((1,)))

        self.ln_1 = nn.LayerNorm((num_nodes), elementwise_affine=False)
        self.ln_2 = nn.LayerNorm((num_nodes), elementwise_affine=False)

        self.reset_parameters(lidx)

    def reset_parameters(self, lidx, how='layerwise'):
        # initialize params
        if how == 'normal':
            nn.init.normal_(self.tau_1)
            nn.init.normal_(self.tau_2)
        else:
            nn.init.constant_(self.tau_1, 1 / (lidx + 1))
            nn.init.constant_(self.tau_2, 1 / (lidx + 1))
        return

    def forward(self, g, h, logits, old_z, shared_tau=True, tau_1=None, tau_2=None):
        # operates on a node
        g = g.local_var()
        if self.dropout:
            h = self.dropout(h)
        g.ndata['h'] = h
        g.ndata['logits'] = logits

        g.update_all(message_func=fn.copy_u('logits', 'logits'), reduce_func=adaptive_reduce_func)
        f1 = g.ndata.pop('f1')
        f2 = g.ndata.pop('f2')
        norm_f1 = self.ln_1(f1)
        norm_f2 = self.ln_2(f2)
        if shared_tau:
            z = F.sigmoid((-1) * (norm_f1 - tau_1)) * F.sigmoid((-1) * (norm_f2 - tau_2))
        else:
            # tau for each layer
            z = F.sigmoid((-1) * (norm_f1 - self.tau_1)) * F.sigmoid((-1) * (norm_f2 - self.tau_2))

        gate = torch.min(old_z, z)
        g.update_all(message_func=fn.copy_u('h', 'feat'), reduce_func=fn.sum(msg='feat', out='agg'))

        agg = g.ndata.pop('agg')

        # normagg = agg * g.ndata['norm']  # normalization by tgt degree
        normagg = agg
        if self.activation:
            normagg = self.activation(normagg)
        new_h = h + gate.unsqueeze(1) * normagg
        return new_h, z


class AdaptiveGNN(nn.Module):
    def __init__(self, model_config):
        super(AdaptiveGNN, self).__init__()
        in_feats = model_config['in_feats']
        n_hidden = model_config['n_hidden']
        n_classes = model_config['n_classes']
        n_layers = model_config['n_layers']
        dropout = model_config['dropout']
        shared_tau = True
        node_each_graph = model_config['node_each_graph']

        self.n_classes = n_classes
        self.init_weight_y = nn.Linear(in_feats, n_classes)
        self.weight_y = nn.Linear(n_hidden, n_classes)

        self.global_tau_1 = nn.Parameter(torch.zeros((1,)))
        self.global_tau_2 = nn.Parameter(torch.zeros((1,)))
        self.shared_tau = shared_tau

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_feats, n_hidden, None, 0.))
        for i in range(n_layers-1):
            self.layers.append(GatedLayer(n_hidden, n_hidden, None, dropout, node_each_graph, i+1))
        self.graph_classification_liner_layer = nn.Linear(n_hidden * node_each_graph, n_classes)

    # def init_weight(self, feats, labels):
    #     A = torch.mm(feats.t(), feats) + 1e-05 * torch.eye(feats.size(1))
    #     labels_one_hot = torch.zeros((feats.size(0), self.n_classes))
    #     for i in range(labels.size(0)):
    #         l = labels[i]
    #         labels_one_hot[i,l] = 1
    #
    #     self.init_weight_y = nn.Parameter(torch.mm(torch.mm(torch.cholesky_inverse(A),feats.t()),labels_one_hot),requires_grad=False)
    #     nn.init.constant_(self.global_tau_1, 1/2)
    #     nn.init.constant_(self.global_tau_2, 1/2)
    #     return

    def forward(self, graph):
        z = torch.FloatTensor([1.0,])
        h = graph.ndata['feat'].to(torch.float32)
        list_z = []
        for lidx, layer in enumerate(self.layers):
            if lidx == 0:
                # logits = F.softmax(self.init_weight_y(h), dim=1)
                # logits = F.softmax(torch.mm(h,self.init_weight_y), dim=1)
                h = layer(graph, h)
            else:
                logits = F.softmax(self.weight_y(h), dim=1)
                h, z = layer(graph, h, logits, old_z=z, shared_tau=self.shared_tau, tau_1=self.global_tau_1, tau_2=self.global_tau_2)
                list_z.append(z)
        h = h.reshape(1, -1)
        logits = self.graph_classification_liner_layer(h)
        return logits
        # output = self.weight_y(h)
        # all_z = torch.stack(list_z, dim=1) # (n_nodes, n_layers)
        # return output, all_z


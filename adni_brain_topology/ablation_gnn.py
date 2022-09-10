import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from torch.distributions import Bernoulli
import dgl
from sklearn.metrics import accuracy_score, f1_score

from utils import record_configuration
from ablation_model import GCN, GIN, GAT, VGAE, VGAE_V2
from data_preprocess import load_pkl
from utils import seed_setting, get_summary_writer


# 数据增强手段：删边
class DropEdge():
    def __init__(self, p=0.5):
        self.p = p
        self.dist = Bernoulli(p)

    def __call__(self, g):
        if self.p == 0:
            return g

        for c_etype in g.canonical_etypes:
            samples = self.dist.sample(torch.Size([g.num_edges(c_etype)]))
            eids_to_remove = g.edges(form='eid', etype=c_etype)[samples.bool().to(g.device)]
            g.remove_edges(eids_to_remove, etype=c_etype)
        return g


# 数据增强手段
def graph_augment(graph, drop_edge_ratio):
    transformer = DropEdge(p=drop_edge_ratio)
    augmented_graph = transformer(graph)
    return augmented_graph


def build_model(model_config):
    if model_config['model_name'] == 'GCN':
        model = GCN(model_config)
    elif model_config['model_name'] == 'GIN':
        model = GIN(model_config)
    elif model_config['model_name'] == 'GraphCL':
        model = GCN(model_config)
    elif model_config['model_name'] == 'GAT':
        model = GAT(model_config)
    elif model_config['model_name'] == 'VGAE':
        model = VGAE_V2(model_config)
    else:
        raise NotImplementedError
    return model


# def train_and_eval_unsupervised_contrastive(train_test_config, dataset_config, model_config, summary_writer):
#     # pretune
#     n_cv = train_test_config['n_cv']
#     n_epoch = train_test_config['n_epoch']
#     n_ss_epoch = train_test_config['n_ss_epoch']
#     dataloader_dir = dataset_config['dataloader_dir']
#     unaug_loader_type = dataset_config['unaug_loader_type']
#     drop_edge_ratio = dataset_config['drop_edge_ratio']
#     for each_cv in range(n_cv):
#         model = build_model(model_config)
#         loss_func = nn.CrossEntropyLoss()
#         contrastive_loss_func = NTXentLoss()
#         optimizer = optim.Adam(model.parameters(), lr=train_test_config['lr'], weight_decay=train_test_config['weight_decay'])
#         self_supervised_optimizer = optim.Adam(model.parameters(), lr=train_test_config['ss_lr'], weight_decay=train_test_config['ss_weight_decay'])
#
#         # for unsupervised train
#         for each_epoch in range(n_ss_epoch):
#             train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
#             loss_value = 0
#             for i in range(len(train_dataset_each_epoch)):
#                 batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
#                 aug_1_batch_graph = graph_augment(batch_graph, drop_edge_ratio=drop_edge_ratio)
#                 aug_2_batch_graph = graph_augment(batch_graph, drop_edge_ratio=drop_edge_ratio)
#                 aug_1_embedding = model.get_embedding(aug_1_batch_graph, batch_size)
#                 aug_2_embedding = model.get_embedding(aug_2_batch_graph, batch_size)
#                 assert aug_1_embedding.size(0) == aug_2_embedding.size(0)
#                 embeddings = torch.cat((aug_1_embedding, aug_2_embedding))
#                 indices = torch.arange(aug_1_embedding.size(0))
#                 label = torch.cat((indices, indices))
#                 contrastive_loss = contrastive_loss_func(embeddings, label)
#                 self_supervised_optimizer.zero_grad()
#                 contrastive_loss.backward()
#                 self_supervised_optimizer.step()
#                 loss_value = loss_value + contrastive_loss.item()
#             print('contrastive loss ' + str(each_epoch) + " " + str(loss_value))
#             summary_writer.add_scalar('generative loss', loss_value, each_epoch)
#
#
#         # for supervised train & evaluation
#         for each_epoch in range(n_epoch):
#             train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
#             test_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_test.pkl'))
#             loss_value = 0
#             for i in range(len(train_dataset_each_epoch)):
#                 batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
#                 logits = model(batch_graph, batch_size)
#                 loss = loss_func(logits, torch.tensor(batch_label))
#                 loss_value = loss_value + loss.item()
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             with torch.no_grad():
#                 for i in range(len(test_dataset_each_epoch)):
#                     batch_graph, batch_label, batch_size = generate_batch_graph_label(test_dataset_each_epoch[i])
#                     logits = model(batch_graph, batch_size)
#                     batch_pred = get_batch_pred(logits)
#                     if i == 0:
#                         pred_each_epoch = batch_pred
#                         label_each_epoch = batch_label
#                     else:
#                         pred_each_epoch = pred_each_epoch + batch_pred
#                         label_each_epoch = label_each_epoch + batch_label
#                 acc = accuracy_score(pred_each_epoch, label_each_epoch)
#                 f1 = f1_score(pred_each_epoch, label_each_epoch, average='weighted')
#             print(each_epoch, loss_value, acc, f1)
#             summary_writer.add_scalar('loss', loss_value, each_epoch)
#             summary_writer.add_scalar('acc', acc, each_epoch)
#             summary_writer.add_scalar('f1', f1, each_epoch)

def train_and_eval_unsupervised_contrastive(train_test_config, dataset_config, model_config, summary_writer):
    # pretune
    n_cv = train_test_config['n_cv']
    n_epoch = train_test_config['n_epoch']
    n_ss_epoch = train_test_config['n_ss_epoch']
    dataloader_dir = dataset_config['dataloader_dir']
    unaug_loader_type = dataset_config['unaug_loader_type']
    drop_edge_ratio = dataset_config['drop_edge_ratio']

    for each_cv in range(n_cv):
        model = build_model(model_config)
        loss_func = nn.CrossEntropyLoss()
        contrastive_loss_func = NTXentLoss()
        optimizer = optim.Adam(model.parameters(), lr=train_test_config['lr'], weight_decay=train_test_config['weight_decay'])
        # for unsupervised train
        for each_epoch in range(n_ss_epoch):
            # # 固定参数
            # for k, v in model.named_parameters():
            #     if k == 'classification.bias' or k == 'classification.weight':
            #         v.requires_grad = False
            loss_value = 0
            train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))

            for i in range(len(train_dataset_each_epoch)):
                batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
                aug_1_batch_graph = graph_augment(batch_graph, drop_edge_ratio=drop_edge_ratio)
                aug_2_batch_graph = graph_augment(batch_graph, drop_edge_ratio=drop_edge_ratio)
                aug_1_embedding = model.get_embedding(aug_1_batch_graph, batch_size)
                aug_2_embedding = model.get_embedding(aug_2_batch_graph, batch_size)
                assert aug_1_embedding.size(0) == aug_2_embedding.size(0)
                embeddings = torch.cat((aug_1_embedding, aug_2_embedding))
                indices = torch.arange(aug_1_embedding.size(0))
                label = torch.cat((indices, indices))
                contrastive_loss = contrastive_loss_func(embeddings, label)
                optimizer.zero_grad()
                contrastive_loss.backward()
                optimizer.step()
                loss_value = loss_value + contrastive_loss.item()
            print('contrastive loss ' + str(each_epoch) + " " + str(loss_value))
            summary_writer.add_scalar('generative loss', loss_value, each_epoch)

        # for supervised train & evaluation
        for each_epoch in range(n_epoch):
            # for k, v in model.named_parameters():
            #     v.requires_grad = False
            #     if k == 'classification.bias' or k == 'classification.weight':
            #         v.requires_grad = True
            loss_value = 0
            train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            test_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_test.pkl'))
            for i in range(len(train_dataset_each_epoch)):
                batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
                logits = model(batch_graph, batch_size)
                loss = loss_func(logits, torch.tensor(batch_label))
                loss_value = loss_value + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for i in range(len(test_dataset_each_epoch)):
                    batch_graph, batch_label, batch_size = generate_batch_graph_label(test_dataset_each_epoch[i])
                    logits = model(batch_graph, batch_size)
                    batch_pred = get_batch_pred(logits)
                    if i == 0:
                        pred_each_epoch = batch_pred
                        label_each_epoch = batch_label
                    else:
                        pred_each_epoch = pred_each_epoch + batch_pred
                        label_each_epoch = label_each_epoch + batch_label
                acc = accuracy_score(pred_each_epoch, label_each_epoch)
                f1 = f1_score(pred_each_epoch, label_each_epoch, average='weighted')
            print(each_epoch, loss_value, acc, f1)
            summary_writer.add_scalar('loss', loss_value, each_epoch)
            summary_writer.add_scalar('acc', acc, each_epoch)
            summary_writer.add_scalar('f1', f1, each_epoch)


def train_and_eval_unsupervised_generative(train_test_config, dataset_config, model_config, summary_writer):
    n_cv = train_test_config['n_cv']
    n_epoch = train_test_config['n_epoch']
    n_ss_epoch = train_test_config['n_ss_epoch']
    dataloader_dir = dataset_config['dataloader_dir']
    unaug_loader_type = dataset_config['unaug_loader_type']

    for each_cv in range(n_cv):
        model = build_model(model_config)
        loss_func = nn.CrossEntropyLoss()
        self_supervised_optimizer = optim.Adam(model.parameters(), lr=train_test_config['ss_lr'],
                                               weight_decay=train_test_config['ss_weight_decay'])
        optimizer = optim.Adam(model.parameters(), lr=train_test_config['lr'],
                               weight_decay=train_test_config['weight_decay'])

        # for unsupervised train
        for each_epoch in range(n_ss_epoch):
            # 固定参数
            for k, v in model.named_parameters():
                if k == 'classification.bias' or k == 'classification.weight':
                    v.requires_grad = False
            loss_value = 0
            train_dataset_each_epoch = load_pkl(
                os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))

            for i in range(len(train_dataset_each_epoch)):
                batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
                adj = batch_graph.adjacency_matrix().to_dense()
                batch_graph = dgl.add_self_loop(batch_graph)
                reconstruct_adj = model.encode_decode(batch_graph)
                loss = F.binary_cross_entropy(adj, reconstruct_adj)
                loss_value = loss_value + loss.item()
                self_supervised_optimizer.zero_grad()
                loss.backward()
                self_supervised_optimizer.step()
            print('generative loss ' + str(each_epoch) + " " + str(loss_value))
            summary_writer.add_scalar('generative loss', loss_value, each_epoch)

        # for supervised train & evaluation
        for each_epoch in range(n_epoch):
            train_dataset_each_epoch = load_pkl(
                os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            test_dataset_each_epoch = load_pkl(
                os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_test.pkl'))
            # for train
            loss_value = 0.0
            for k, v in model.named_parameters():
                v.requires_grad = False
                if k == 'classification.bias' or k == 'classification.weight':
                    v.requires_grad = True

            for i in range(len(train_dataset_each_epoch)):
                batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
                logits = model(batch_graph, batch_size)
                loss = loss_func(logits, torch.tensor(batch_label))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = loss_value + loss.item()

            # for evaluation
            with torch.no_grad():
                for i in range(len(test_dataset_each_epoch)):
                    batch_graph, batch_label, batch_size = generate_batch_graph_label(test_dataset_each_epoch[i])
                    logits = model(batch_graph, batch_size)
                    batch_pred = get_batch_pred(logits)
                    if i == 0:
                        pred_each_epoch = batch_pred
                        label_each_epoch = batch_label
                    else:
                        pred_each_epoch = pred_each_epoch + batch_pred
                        label_each_epoch = label_each_epoch + batch_label
                acc = accuracy_score(pred_each_epoch, label_each_epoch)
                f1 = f1_score(pred_each_epoch, label_each_epoch, average='weighted')
            print(each_epoch, loss_value, acc, f1)
            summary_writer.add_scalar('loss', loss_value, each_epoch)
            summary_writer.add_scalar('acc', acc, each_epoch)
            summary_writer.add_scalar('f1', f1, each_epoch)


# 有监督的
def train_and_eval_supervised(train_test_config, dataset_config, model_config, summary_writer):
    n_cv = train_test_config['n_cv']
    n_epoch = train_test_config['n_epoch']
    dataloader_dir = dataset_config['dataloader_dir']
    unaug_loader_type = dataset_config['unaug_loader_type']
    for each_cv in range(n_cv):
        model = build_model(model_config)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=train_test_config['lr'], weight_decay=train_test_config['weight_decay'])
        for each_epoch in range(n_epoch):
            train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            test_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_test.pkl'))

            loss_value = 0
            # for train
            for i in range(len(train_dataset_each_epoch)):
                batch_graph, batch_label, batch_size = generate_batch_graph_label(train_dataset_each_epoch[i])
                logits = model(batch_graph, batch_size)
                loss = loss_func(logits, torch.tensor(batch_label))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = loss_value + loss.item()

            # for evaluation
            with torch.no_grad():
                pred_each_epoch = []
                label_each_epoch = []
                for i in range(len(test_dataset_each_epoch)):
                    batch_graph, batch_label, batch_size = generate_batch_graph_label(test_dataset_each_epoch[i])
                    logits = model(batch_graph, batch_size)
                    batch_pred = get_batch_pred(logits)
                    if i == 0:
                        pred_each_epoch = batch_pred
                        label_each_epoch = batch_label
                    else:
                        pred_each_epoch = pred_each_epoch + batch_pred
                        label_each_epoch = label_each_epoch + batch_label
                acc = accuracy_score(pred_each_epoch, label_each_epoch)
                f1 = f1_score(pred_each_epoch, label_each_epoch, average='weighted')
            print(each_epoch, loss_value, acc, f1)
            summary_writer.add_scalar('loss', loss_value, each_epoch)
            summary_writer.add_scalar('acc', acc, each_epoch)
            summary_writer.add_scalar('f1', f1, each_epoch)


def get_batch_pred(logits):
    return logits.argmax(1).tolist()


def generate_batch_graph_label(batch_graph_label):
    batch_graph_list = []
    batch_label_list = []
    for each_batch_graph_label in batch_graph_label:
        batch_graph_list.append(each_batch_graph_label[0])
        batch_label_list.append(each_batch_graph_label[1])
    return dgl.batch(batch_graph_list), batch_label_list, len(batch_graph_label)


def train_and_eval(train_test_config, dataset_config, model_config, summary_writer):
    if model_config['model_name'] in ['GCN', 'GAT', 'GIN']:
        train_and_eval_supervised(train_test_config, dataset_config, model_config, summary_writer)
    elif model_config['model_name'] in ['VGAE']:
        train_and_eval_unsupervised_generative(train_test_config, dataset_config, model_config, summary_writer)
    else:
        train_and_eval_unsupervised_contrastive(train_test_config, dataset_config, model_config, summary_writer)


def main():
    seed_setting(1024)
    train_test_config = {
        'n_cv': 1,
        'n_ss_epoch': 40,
        'n_epoch': 200,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'ss_lr': 1e-5,
        'ss_weight_decay': 1e-3,
        'comment': 'binary & without freezing parameters',
    }
    dataset_config = {
        # 'dataloader_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/dataloader',
        'dataloader_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/dataloader_binary',
        'unaug_loader_type': 'NoAug_NoAug',
        'drop_edge_ratio': 0.2
    }
    model_config = {
        'model_name': 'GraphCL',

        'in_dim': 246,
        'hid_dim': 1024,
        'n_classes': 2,
        'activation': F.relu,
        'node_each_graph': 246,
        'n_layers': 2,
        'in_feats': 246,
        'dropout': 0.2,

        'learn_eps': True,
        'n_mlp_layers': 2,
        'neighbor_pooling_type': "sum",
        'graph_pooling_type': "sum",
        'final_dropout': 0.5,
        
        'heads': [8,1],
        'feat_drop': 0.6,
        'attn_drop': 0.6,
        'negative_slope': 0.2,
        'residual': False,

        'mask_ratio': 0.2,
    }
    summary_writer, log_dir = get_summary_writer('run')
    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_test_config,
    })

    train_and_eval(train_test_config, dataset_config, model_config, summary_writer)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()

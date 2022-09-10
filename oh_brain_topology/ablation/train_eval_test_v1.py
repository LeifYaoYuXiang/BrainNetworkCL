import torch
import numpy as np
from scipy.sparse import coo_matrix
from dgl.sampling import node2vec_random_walk
from sklearn.metrics import accuracy_score, f1_score
from dgl import DropEdge, FeatMask


def dropping_node_graph_augmentation(each_graph, dropping_node_ratio=0.15, device='cpu'):
    n_feat = each_graph.ndata['feat']
    masks = torch.bernoulli(1. - torch.FloatTensor(np.ones(n_feat.shape[0]) * dropping_node_ratio)).unsqueeze(1)
    if device == 'gpu':
        masks = masks.cuda()
    n_feat = masks * n_feat
    aug_each_graph = each_graph
    aug_each_graph.ndata['feat'] = n_feat
    return aug_each_graph


def dropping_edge_graph_augmentation(each_graph, drop_edge_ratio=0.15, device='cpu'):
    transform = DropEdge(drop_edge_ratio)
    aug_each_graph = transform(each_graph)
    return aug_each_graph


def subgraph_generate_augmentation(each_graph, walk_length=10, device='cpu'):
    n_node = each_graph.ndata['feat'].shape[0]
    end_index = node2vec_random_walk(each_graph.to('cpu'), list(range(n_node)), 1, 1, walk_length=walk_length, return_eids=True)[0]
    end_index = end_index.flatten()
    start_index = torch.LongTensor(list(range(n_node)))
    start_index = start_index.repeat_interleave(walk_length + 1)
    matrix = coo_matrix(([1] * start_index.shape[0], (start_index, end_index)), shape=(n_node, n_node), dtype=np.int).todense()
    aug_each_graph = each_graph
    if device == 'cpu':
        aug_each_graph.ndata['feat'] = torch.matmul(torch.Tensor(matrix).double(), aug_each_graph.ndata['feat'])
    else:
        aug_each_graph.ndata['feat'] = torch.matmul(torch.Tensor(matrix).double().cuda(), aug_each_graph.ndata['feat'])
    return aug_each_graph


def feature_masking_graph_augmentation(each_graph, feature_masking_ratio=0.15, device='cpu'):
    transform = FeatMask(feature_masking_ratio, node_feat_names=['feat'])
    aug_each_graph = transform(each_graph)
    return aug_each_graph


def g_mixup_graph_augmentation(each_graph):
    pass


aug_methods_dict = {
    'dropping_node': dropping_node_graph_augmentation,
    'dropping_edge': dropping_edge_graph_augmentation,
    'subgraph_generation': subgraph_generate_augmentation,
    'feat_masking': feature_masking_graph_augmentation,
    'g_mixup': g_mixup_graph_augmentation,
}


def train_test_eval_v1(train_dataloader, eval_dataloader,
                    model, cl_optimizer, optimizer, cl_loss_fcn, loss_fcn, cl_scheduler, scheduler,
                    train_test_config, summary_writer):
    cl_n_epoch = train_test_config['cl_n_epoch']
    n_epoch = train_test_config['n_epoch']
    aug_1_method = aug_methods_dict[train_test_config['aug_1_method']]
    aug_2_method = aug_methods_dict[train_test_config['aug_2_method']]
    batch_size = train_test_config['batch_size']
    device = train_test_config['device']

    # contrastive learning for pretraining
    for each_epoch in range(cl_n_epoch):
        # train the model
        cl_loss_val = 0.0
        train_data = train_dataloader.get_batch(batch_size)
        for batch_graph_label in train_data:
            batch_index = 0
            for each_graph_label in batch_graph_label:
                # batch_index = 0
                each_graph = each_graph_label[0]
                # each_label = each_graph_label[1]
                aug_1_graph = aug_1_method(each_graph, device=device)
                aug_2_graph = aug_2_method(each_graph, device=device)
                if batch_index == 0:
                    batch_graph_embedding_aug_1 = model.get_embedding(aug_1_graph)
                    batch_graph_embedding_aug_2 = model.get_embedding(aug_2_graph)
                else:
                    batch_graph_embedding_aug_1 = torch.concat((batch_graph_embedding_aug_1, model.get_embedding(aug_1_graph)), 0)
                    batch_graph_embedding_aug_2 = torch.concat((batch_graph_embedding_aug_2, model.get_embedding(aug_2_graph)), 0)
                batch_index = batch_index + 1
            # cl_loss = cl_loss_fcn(batch_graph_embedding_aug_1, batch_graph_embedding_aug_2)
            aug_embeddings = torch.cat((batch_graph_embedding_aug_1, batch_graph_embedding_aug_2))
            indices = torch.arange(batch_index)
            if device == 'cpu':
                label = torch.cat((indices, indices))
            else:
                label = torch.cat((indices, indices)).cuda()
            cl_loss = cl_loss_fcn(aug_embeddings, label)
            cl_loss_val = cl_loss_val + cl_loss.item()
            cl_optimizer.zero_grad()
            cl_loss.backward()
            cl_optimizer.step()
        cl_scheduler.step()
        summary_writer.add_scalar('cl_loss', cl_loss_val, each_epoch)
        print('contrastive learning', each_epoch, cl_loss_val)

    # train and evaluation
    for each_epoch in range(n_epoch):
        # train the model
        labeled_loss_value = 0.0
        for train_batch_graph_label in train_dataloader.get_batch(batch_size):
            train_batch_index = 0
            train_batch_label = []
            for each_graph_label in train_batch_graph_label:
                each_graph = each_graph_label[0]
                each_label = each_graph_label[1]
                if train_batch_index == 0:
                    batch_graph_logits = model(each_graph)
                else:
                    batch_graph_logits = torch.concat((batch_graph_logits, model(each_graph)), 0)
                train_batch_label.append(each_label)
                train_batch_index = train_batch_index + 1
            if device == 'cpu':
                train_batch_label = torch.LongTensor(train_batch_label)
            else:
                train_batch_label = torch.LongTensor(train_batch_label).cuda()
            label_loss = loss_fcn(batch_graph_logits, train_batch_label)
            optimizer.zero_grad()
            label_loss.backward()
            optimizer.step()
            labeled_loss_value = labeled_loss_value + label_loss.item()

        # eval the model
        with torch.no_grad():
            eval_epoch_logits = []
            eval_epoch_label = []
            for eval_batch_graph_label in eval_dataloader.get_batch(batch_size):
                eval_batch_index = 0
                eval_batch_label = []
                for each_graph_label in eval_batch_graph_label:
                    each_graph = each_graph_label[0]
                    each_label = each_graph_label[1]
                    if eval_batch_index == 0:
                        batch_graph_logits = model(each_graph)
                    else:
                        batch_graph_logits = torch.concat((batch_graph_logits, model(each_graph)), 0)
                    eval_batch_label.append(each_label)
                    eval_batch_index = eval_batch_index + 1
                eval_batch_logits = batch_graph_logits.argmax(1).tolist()
                eval_epoch_logits = eval_epoch_logits + eval_batch_logits
                eval_epoch_label = eval_epoch_label + eval_batch_label
        # print(eval_epoch_logits, eval_epoch_label)
        acc = accuracy_score(eval_epoch_logits, eval_epoch_label)
        f1 = f1_score(eval_epoch_logits, eval_epoch_label)
        summary_writer.add_scalar('labeled_loss', labeled_loss_value, each_epoch)
        summary_writer.add_scalar('Acc', acc, each_epoch)
        summary_writer.add_scalar('F1', f1, each_epoch)
        print('labeled learning', each_epoch, labeled_loss_value, acc, f1)



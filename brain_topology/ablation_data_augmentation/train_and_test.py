import os
import numpy as np
import dgl
import torch
import random
from metrics import precision_metric, recall_metric, acc_metric, f1_metric
from utils import write_list_to_file


def train_and_eval_ablation(train_test_config, dataloader_cv_dict, model, ml_model, loss_fcn, optimizer, summary_writer, log_dir):
    n_epoch = train_test_config['n_epoch']
    drop_ratio = 0.15
    # 每一折的训练和测试
    for each_exp in range(train_test_config['cv_number']):
        print('第' + str(each_exp + 1) + '折数据训练开始')
        train_dataloader_this_cv = dataloader_cv_dict[each_exp][0]
        test_dataloader_this_cv = dataloader_cv_dict[each_exp][1]

        acc_list = []
        f1_list = []
        # 每一次的训练和测试
        for n in range(n_epoch):
            train_dataloader_this_epoch = train_dataloader_this_cv[n]
            test_dataloader_this_epoch = test_dataloader_this_cv[n]

            dataloader_length = len(train_dataloader_this_epoch)
            ## train GNN Encoder
            # 采用 “普通” 的手段对数据进行增强
            # aug_graph_list_1 = generate_augmentation_drop_edge(train_dataloader_this_epoch, dataloader_length, drop_ratio)
            # aug_graph_list_2 = generate_augmentation_drop_edge(train_dataloader_this_epoch, dataloader_length, drop_ratio)
            aug_graph_list_1 = generate_augmentation_drop_node(train_dataloader_this_epoch, dataloader_length, drop_ratio)
            aug_graph_list_2 = generate_augmentation_drop_node(train_dataloader_this_epoch, dataloader_length, drop_ratio)
            bg_1 = dgl.batch(aug_graph_list_1)
            bg_2 = dgl.batch(aug_graph_list_2)
            batch_size = int(bg_1.ndata['feat'].shape[0] / bg_1.ndata['feat'].shape[1])
            # 开始训练
            logits1 = model((bg_1.ndata['feat'].to(torch.float32), bg_1, batch_size))
            logits2 = model((bg_2.ndata['feat'].to(torch.float32), bg_2, batch_size))
            total_node_number = logits1.size(0)
            embeddings = torch.cat((logits1, logits2))
            indices = torch.arange(total_node_number)
            label = torch.cat((indices, indices))
            loss = loss_fcn(embeddings, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## train ML model
            model.eval()
            ml_train_dataloader = train_dataloader_this_epoch
            with torch.no_grad():
                for i in range(len(ml_train_dataloader)):
                    graph_info = ml_train_dataloader[i]
                    graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
                    batch_size = graph_info['batch_size']
                    graph = graph_info['batch_graph']
                    graph = dgl.add_self_loop(graph)
                    input = (graph_node_features, graph, batch_size)
                    embedding = model.get_embedding(input)
                    if i == 0:
                        data = embedding.numpy()
                        label = np.array(graph_info['batch_label'])
                    else:
                        data = np.vstack((data, embedding.numpy()))
                        label = np.append(label, np.array(graph_info['batch_label']))
                ml_model.fit(data, label)

                ## 进行测试
                acc, f1 = eval_using_ml(model, ml_model, test_dataloader_this_epoch)
                # 输出结果
                print('Epoch:', n, loss.item(), acc, f1)
                summary_writer.add_scalar(str(each_exp) + '/Loss', loss.item(), n)
                summary_writer.add_scalar(str(each_exp) + '/Acc', acc, n)
                summary_writer.add_scalar(str(each_exp) + '/F1', f1, n)
                acc_list.append(acc)
                f1_list.append(f1)

        # 将每一折的训练和测试的结果进行记录
        write_list_to_file(os.path.join(log_dir, str(each_exp) + '_acc.txt'), acc_list)
        write_list_to_file(os.path.join(log_dir, str(each_exp) + '_f1.txt'), f1_list)
        print('ACC AVG:', np.array(acc_list).mean())
        print('F1 AVG:', np.array(f1_list).mean())


# return one augmentation graph list by dropping edges
def generate_augmentation_drop_edge(graph_list, dataloader_length, drop_ratio):
    augmentation_batch_graph_list = []
    for i in range(dataloader_length):
        each_batch_graph = graph_list[i]['batch_graph']
        select_edge_number = int(each_batch_graph.num_edges() * (drop_ratio))
        random_select_edge = random.sample(range(each_batch_graph.num_edges()), select_edge_number)
        aug_graph = dgl.remove_edges(each_batch_graph, torch.tensor(random_select_edge))
        augmentation_batch_graph_list.append(aug_graph)
    return augmentation_batch_graph_list


# return one augmentation graph list by dropping nodes
def generate_augmentation_drop_node(graph_list, dataloader_length, drop_ratio):
    augmentation_batch_graph_list = []
    for i in range(dataloader_length):
        each_batch_graph = graph_list[i]['batch_graph']
        original_graph_feats = each_batch_graph.ndata['feat']
        masks = torch.bernoulli(1. - torch.FloatTensor(np.ones(original_graph_feats.shape[0]) * drop_ratio)).unsqueeze(1)
        graph_feats = masks * original_graph_feats
        each_batch_graph.ndata['feat'] = graph_feats
        augmentation_batch_graph_list.append(each_batch_graph)
    return augmentation_batch_graph_list




def eval_using_ml(model, ml_model, test_loader):
    model.eval()
    with torch.no_grad():
        for i in range(len(test_loader)):
            graph_info = test_loader[i]
            graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
            batch_size = graph_info['batch_size']
            graph = graph_info['batch_graph']
            graph_batch_label = torch.from_numpy(np.array(graph_info['batch_label']))
            graph = dgl.add_self_loop(graph)
            input = (graph_node_features, graph, batch_size)
            embedding = model.get_embedding(input)
            logits = torch.from_numpy(ml_model.predict(embedding.numpy()))
            if i == 0:
                indices_record = logits
                batch_y_record = graph_batch_label
            else:
                indices_record = torch.cat((indices_record, logits), 0)
                batch_y_record = torch.cat((batch_y_record, graph_batch_label), 0)
        acc = acc_metric(indices_record, batch_y_record)
        f1 = f1_metric(precision=precision_metric(indices_record, batch_y_record), recall=recall_metric(indices_record, batch_y_record))
        return acc, f1
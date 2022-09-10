import os
import numpy as np
import torch
import dgl
from pytorch_metric_learning.losses import NTXentLoss
from sklearn.linear_model import LogisticRegression
from data_process.data_preprocess import load_data_loaders_from_pkl
from metrics import precision_metric, recall_metric, acc_metric, f1_metric
from model_v_gin import Model
from utils import save_model, write_list_to_file


def train_and_eval(train_test_config, dataset_config, model_config, summary_writer, log_dir):
    n_epoch = train_test_config['n_epoch']

    for each_exp in range(train_test_config['cv_number']):
        print('第'+str(each_exp+1)+'折数据生成开始')

        # 生成每一折用来训练和测试的模型、优化器、损失函数
        model = Model(model_config)
        ml_model = LogisticRegression()
        loss_fcn = NTXentLoss(temperature=train_test_config['nt_xent_loss_temperature'])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])

        # 生成每一折用来训练和测试的数据
        dataloader_dir = dataset_config['dataloader_dir']
        train1_loader_type = dataset_config['train1_loader_type']
        train2_loader_type = dataset_config['train2_loader_type']
        unaug_loader_type = dataset_config['unaug_loader_type']
        train_loader1_path_list = []
        train_loader1_path_this_cv = os.path.join(dataloader_dir, train1_loader_type, str(each_exp), 'train')
        for loader in os.listdir(train_loader1_path_this_cv):
            train_loader1_path_list.append(os.path.join(train_loader1_path_this_cv, loader))

        train_loader2_path_list = []
        train_loader2_path_this_cv = os.path.join(dataloader_dir, train2_loader_type, str(each_exp), 'train')
        for loader in os.listdir(train_loader2_path_this_cv):
            train_loader2_path_list.append(os.path.join(train_loader2_path_this_cv, loader))

        unaug_train_path_list = []
        unaug_train_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(each_exp), 'train')
        for loader in os.listdir(unaug_train_loader_path_this_cv):
            unaug_train_path_list.append(os.path.join(unaug_train_loader_path_this_cv, loader))

        unaug_test_path_list = []
        unaug_test_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(each_exp), 'test')
        for loader in os.listdir(unaug_test_loader_path_this_cv):
            unaug_test_path_list.append(os.path.join(unaug_test_loader_path_this_cv, loader))

        # 对每一折进行预训练
        model, ml_model, acc_list, f1_list = train_and_eval_each_split(each_exp, n_epoch, model, ml_model, loss_fcn, optimizer,
                                          train_loader1_path_list, train_loader2_path_list,
                                          unaug_train_path_list, unaug_test_path_list, summary_writer)
        # save_model(model, os.path.join(log_dir, str(each_exp)+'_cv.model.pkl'))
        write_list_to_file(os.path.join(log_dir, str(each_exp) + '_acc.txt'), acc_list)
        write_list_to_file(os.path.join(log_dir, str(each_exp) + '_f1.txt'), f1_list)
        print('ACC AVG:', np.array(acc_list).mean())
        print('F1 AVG:', np.array(f1_list).mean())


def train_and_eval_each_split(each_exp, n_epoch, model, ml_model, loss_fcn, optimizer,
                              train_loader1_path_list, train_loader2_path_list,
                              unaug_train_loader_path_list, unaug_test_path_loader_list, summary_writer):
    acc_list = []
    f1_list = []
    for n in range(n_epoch):
        train_loader_path_this_epoch = train_loader1_path_list[n]
        train_loader2_path_this_epoch = train_loader2_path_list[n]
        unaug_train_loader_path_this_epoch = unaug_train_loader_path_list[n]
        unaug_test_loader_path_this_epoch = unaug_test_path_loader_list[n]

        train_loader_this_epoch = load_data_loaders_from_pkl(train_loader_path_this_epoch)
        train_loader2_this_epoch = load_data_loaders_from_pkl(train_loader2_path_this_epoch)
        unaug_train_loader_this_epoch = load_data_loaders_from_pkl(unaug_train_loader_path_this_epoch)
        unaug_test_loader_this_epoch = load_data_loaders_from_pkl(unaug_test_loader_path_this_epoch)

        model.train()
        dataloader_length = len(train_loader_this_epoch)

        bg_1 = dgl.batch([train_loader_this_epoch[i]['batch_graph'] for i in range(dataloader_length)])
        bg_2 = dgl.batch([train_loader2_this_epoch[i]['batch_graph'] for i in range(dataloader_length)])
        batch_size = int(bg_1.ndata['feat'].shape[0] / bg_1.ndata['feat'].shape[1])
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

        # 训练判断器
        model.eval()
        ml_train_dataloader = unaug_train_loader_this_epoch + train_loader_this_epoch + train_loader2_this_epoch
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

        # 进行测试
        acc, f1 = eval_using_ml(model, ml_model, unaug_test_loader_this_epoch)
        # 输出结果
        print('Epoch:', n, loss.item(), acc, f1)
        summary_writer.add_scalar(str(each_exp)+'/Loss', loss.item(), n)
        summary_writer.add_scalar(str(each_exp)+'/Acc', acc, n)
        summary_writer.add_scalar(str(each_exp)+'/F1', f1, n)
        acc_list.append(acc)
        f1_list.append(f1)

    return model, ml_model, acc_list, f1_list



# 用于测试
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

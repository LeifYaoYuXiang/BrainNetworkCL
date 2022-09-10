# for ablation test: without data augmentation and only for logistic regression
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from data_preprocess import load_pkl
from utils import seed_setting
from sklearn.linear_model import LogisticRegression
from metrics import precision_metric, recall_metric, acc_metric, f1_metric


def main():
    train_test_config = {
        'n_epoch': 200,
    }
    dataset_config = {
        'dataloader_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/dataloader',
        'unaug_loader_type': 'NoAug_NoAug',
    }
    seed_setting(1024)
    train_and_eval(train_test_config, dataset_config)


def train_and_eval(train_test_config, dataset_config):
    n_cv = 1
    n_epoch = train_test_config['n_epoch']
    dataloader_dir = dataset_config['dataloader_dir']
    unaug_loader_type = dataset_config['unaug_loader_type']

    for each_cv in range(n_cv):
        ml_model = LogisticRegression()
        for each_epoch in range(n_epoch):
            unaug_train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv)+'_' +str(each_epoch)+'_train.pkl'))
            unaug_test_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv)+'_'+str(each_epoch)+'_test.pkl'))
            ml_dataset = unaug_train_dataset_each_epoch
            # 某一个 epoch中 训练
            data = 0
            label = 0
            for i in range(len(ml_dataset)):
                batch_graph_list, batch_label, batch_size = generate_batch_graph_label(ml_dataset[i])
                embedding = 0
                for j in range(len(batch_graph_list)):
                    each_graph = batch_graph_list[j]
                    feat = each_graph.ndata['feat'].to(torch.float32).reshape(1, -1)
                    if j == 0:
                        embedding = feat
                    else:
                        embedding = torch.cat((embedding, feat))
                if i == 0:
                    data = embedding.numpy()
                    label = np.array(batch_label)
                else:
                    data = np.vstack((data, embedding.numpy()))
                    label = np.append(label, np.array(batch_label))
            ml_model.fit(data, label)

            # 某一个 epoch中 测试
            data = 0
            label = 0
            for i in range(len(unaug_test_dataset_each_epoch)):
                unaug_batch_graph_list, unaug_batch_label, unaug_batch_size = generate_batch_graph_label(unaug_test_dataset_each_epoch[i])
                embedding = 0
                for j in range(len(unaug_batch_graph_list)):
                    each_graph = unaug_batch_graph_list[j]
                    feat = each_graph.ndata['feat'].to(torch.float32).reshape(1, -1)
                    if j == 0:
                        embedding = feat
                    else:
                        embedding = torch.cat((embedding, feat))
                if i == 0:
                    data = embedding.numpy()
                    label = np.array(unaug_batch_label)
                else:
                    data = np.vstack((data, embedding.numpy()))
                    label = np.append(label, np.array(unaug_batch_label))
            logits = ml_model.predict(data)
            acc = accuracy_score(logits, label)
            f1 = f1_score(label, logits, average='weighted')
            # acc = acc_metric(logits, label)
            # f1 = f1_metric(precision=precision_metric(logits, label), recall=recall_metric(logits, label))
            print(each_epoch, acc, f1)


def generate_batch_graph_label(batch_graph_label):
    batch_graph_list = []
    batch_label_list = []
    for each_batch_graph_label in batch_graph_label:
        batch_graph_list.append(each_batch_graph_label[0])
        batch_label_list.append(each_batch_graph_label[1])
    return batch_graph_list, batch_label_list, len(batch_graph_label)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()

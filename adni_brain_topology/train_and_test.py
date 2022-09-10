import os
import numpy as np
import torch
import dgl
from model_v_gin import Model
from data_preprocess import load_pkl
from utils import save_model, write_list_to_file
from sklearn.metrics import accuracy_score, f1_score
from pytorch_metric_learning.losses import NTXentLoss
from sklearn.linear_model import LogisticRegression
from metrics import precision_metric, recall_metric, acc_metric, f1_metric


def train_and_eval(train_test_config, dataset_config, model_config, summary_writer, log_dir):
    n_cv = train_test_config['cv']
    n_cv = 1
    n_epoch = train_test_config['n_epoch']
    dataloader_dir = dataset_config['dataloader_dir']
    train1_loader_type = dataset_config['train1_loader_type']
    train2_loader_type = dataset_config['train2_loader_type']
    unaug_loader_type = dataset_config['unaug_loader_type']

    for each_cv in range(n_cv):
        # create one new model
        model = Model(model_config)
        ml_model = LogisticRegression()
        loss_fcn = NTXentLoss(temperature=train_test_config['nt_xent_loss_temperature'])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])

        # train GNN Encoder
        model.train()
        for each_epoch in range(n_epoch):
            loss_value = 0
            train_aug_1_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, train1_loader_type, str(each_cv)+'_'+str(each_epoch)+'_train.pkl'))
            train_aug_2_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, train2_loader_type, str(each_cv)+'_'+str(each_epoch)+'_train.pkl'))
            unaug_train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv)+'_' +str(each_epoch)+'_train.pkl'))
            unaug_test_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv)+'_'+str(each_epoch)+'_test.pkl'))
            assert len(train_aug_1_dataset_each_epoch) == len(train_aug_2_dataset_each_epoch) == len(unaug_train_dataset_each_epoch)
            dataset_length = len(train_aug_1_dataset_each_epoch)
            # train GNN encoder
            for i in range(dataset_length):
                train_aug_1_batch_graph_label = train_aug_1_dataset_each_epoch[i]
                train_aug_2_batch_graph_label = train_aug_2_dataset_each_epoch[i]
                train_aug_1_batch_graph, train_aug_1_batch_label, train_aug_1_batch_size = generate_batch_graph_label(
                    train_aug_1_batch_graph_label
                )
                train_aug_2_batch_graph, train_aug_2_batch_label, train_aug_2_batch_size = generate_batch_graph_label(
                    train_aug_2_batch_graph_label
                )
                assert train_aug_1_batch_label == train_aug_2_batch_label
                assert train_aug_1_batch_size == train_aug_2_batch_size
                batch_size = train_aug_1_batch_size
                logits1 = model((train_aug_1_batch_graph.ndata['feat'].to(torch.float32), train_aug_1_batch_graph, batch_size))
                logits2 = model((train_aug_2_batch_graph.ndata['feat'].to(torch.float32), train_aug_2_batch_graph, batch_size))
                total_node_number = logits1.size(0)
                embeddings = torch.cat((logits1, logits2))
                indices = torch.arange(total_node_number)
                label = torch.cat((indices, indices))
                loss = loss_fcn(embeddings, label)
                loss_value = loss_value + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # train ML Model
            model.eval()
            with torch.no_grad():
                ml_dataset = train_aug_1_dataset_each_epoch + train_aug_2_dataset_each_epoch + unaug_train_dataset_each_epoch
                for i in range(len(ml_dataset)):
                    batch_graph, batch_label, batch_size = generate_batch_graph_label(
                        ml_dataset[i]
                    )
                    graph_node_features = batch_graph.ndata['feat'].to(torch.float32)
                    batch_graph = dgl.add_self_loop(batch_graph)
                    input = (graph_node_features, batch_graph, batch_size)
                    embedding = model.get_embedding(input)
                    if i == 0:
                        data = embedding.numpy()
                        label = np.array(batch_label)
                    else:
                        data = np.vstack((data, embedding.numpy()))
                        label = np.append(label, np.array(batch_label))
                ml_model.fit(data, label)

            # evaluation
            with torch.no_grad():
                for i in range(len(unaug_test_dataset_each_epoch)):
                    unaug_batch_graph_label = unaug_test_dataset_each_epoch[i]
                    unaug_batch_graph, unaug_batch_label, unaug_batch_size = generate_batch_graph_label(unaug_batch_graph_label)
                    graph_node_features = unaug_batch_graph.ndata['feat'].to(torch.float32)
                    batch_size = unaug_batch_size
                    graph = unaug_batch_graph
                    graph_batch_label = torch.from_numpy(np.array(unaug_batch_label))
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
                f1 = f1_score(batch_y_record.numpy(), indices_record.numpy(), average='weighted')
                # f1 = f1_metric(precision=precision_metric(indices_record, batch_y_record), recall=recall_metric(indices_record, batch_y_record))
                print(each_epoch, loss_value,  acc, f1)
                summary_writer.add_scalar('loss', loss_value, each_epoch)
                summary_writer.add_scalar('Acc', acc, each_epoch)
                summary_writer.add_scalar('F1', f1, each_epoch)


def generate_batch_graph_label(batch_graph_label):
    batch_graph_list = []
    batch_label_list = []
    for each_batch_graph_label in batch_graph_label:
        batch_graph_list.append(each_batch_graph_label[0])
        batch_label_list.append(each_batch_graph_label[1])
    return dgl.batch(batch_graph_list), batch_label_list, len(batch_graph_label)

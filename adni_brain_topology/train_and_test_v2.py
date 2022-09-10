import os
import numpy as np
import torch
import torch.nn as nn
import dgl
from model_v_gin_2 import Model_v2
from model_v_gin_3 import Model_v3
from data_preprocess import load_pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score
from pytorch_metric_learning.losses import NTXentLoss


def train_and_eval_v2(train_test_config, dataset_config, model_config, summary_writer, log_dir):
    n_cv = 1
    n_epoch = train_test_config['n_epoch']
    ss_n_epoch = train_test_config['ss_n_epoch']
    dataloader_dir = dataset_config['dataloader_dir']
    train1_loader_type = dataset_config['train1_loader_type']
    train2_loader_type = dataset_config['train2_loader_type']
    unaug_loader_type = dataset_config['unaug_loader_type']

    for each_cv in range(n_cv):
        # create one new model
        model = Model_v3(model_config)
        contrastive_loss_fcn = NTXentLoss(temperature=train_test_config['nt_xent_loss_temperature'])
        labeled_loss_fcn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        ss_optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['ss_lr'])
        ss_scheduler = StepLR(ss_optimizer, step_size=10, gamma=0.1)
        # for pretune
        for each_epoch in range(ss_n_epoch):
            # 固定参数
            # for k, v in model.named_parameters():
            #     if k == 'fc_layer.weight' or k == 'fc_layer.bias':
            #         v.requires_grad = False
            loss_value = 0
            train_aug_1_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, train1_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            train_aug_2_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, train2_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            unaug_train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))

            assert len(train_aug_1_dataset_each_epoch) == len(train_aug_2_dataset_each_epoch) == len(unaug_train_dataset_each_epoch)
            dataset_length = len(train_aug_1_dataset_each_epoch)

            for i in range(dataset_length):
                train_aug_1_batch_graph_label = train_aug_1_dataset_each_epoch[i]
                train_aug_2_batch_graph_label = train_aug_2_dataset_each_epoch[i]
                train_aug_1_batch_graph, train_aug_1_batch_label, train_aug_1_batch_size = generate_batch_graph_label(train_aug_1_batch_graph_label)
                train_aug_2_batch_graph, train_aug_2_batch_label, train_aug_2_batch_size = generate_batch_graph_label(train_aug_2_batch_graph_label)
                assert train_aug_1_batch_label == train_aug_2_batch_label
                assert train_aug_1_batch_size == train_aug_2_batch_size
                batch_size = train_aug_1_batch_size
                aug_1_embedding = model.get_embedding(train_aug_1_batch_graph, batch_size)
                aug_2_embedding = model.get_embedding(train_aug_2_batch_graph, batch_size)

                embeddings = torch.cat((aug_1_embedding, aug_2_embedding))
                indices = torch.arange(batch_size)
                label = torch.cat((indices, indices))
                loss = contrastive_loss_fcn(embeddings, label)
                loss_value = loss_value + loss.item()
                ss_optimizer.zero_grad()
                loss.backward()
                ss_optimizer.step()
            ss_scheduler.step()
            print(each_epoch, loss_value)

        # for finetune & evaluation
        for each_epoch in range(n_epoch):
            # finetune
            # 固定参数
            # for k, v in model.named_parameters():
            #     v.requires_grad = False
            #     if k == 'fc_layer.weight' or k == 'fc_layer.bias':
            #         v.requires_grad = True
            loss_value = 0
            train_aug_1_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, train1_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            train_aug_2_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, train2_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            unaug_train_dataset_each_epoch = load_pkl(os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_train.pkl'))
            ml_dataset = train_aug_1_dataset_each_epoch + train_aug_2_dataset_each_epoch + unaug_train_dataset_each_epoch

            for i in range(len(ml_dataset)):
                batch_graph, batch_label, batch_size = generate_batch_graph_label(ml_dataset[i])
                logits = model(batch_graph, batch_size)

                labeled_loss = labeled_loss_fcn(logits, torch.LongTensor(batch_label))
                loss_value = loss_value + labeled_loss.item()
                optimizer.zero_grad()
                labeled_loss.backward()
                optimizer.step()
            scheduler.step()
            # evaluation
            with torch.no_grad():
                unaug_test_dataset_each_epoch = load_pkl(
                    os.path.join(dataloader_dir, unaug_loader_type, str(each_cv) + '_' + str(each_epoch) + '_test.pkl'))
                for i in range(len(unaug_test_dataset_each_epoch)):
                    unaug_batch_graph_label = unaug_test_dataset_each_epoch[i]
                    unaug_batch_graph, unaug_batch_label, unaug_batch_size = generate_batch_graph_label(unaug_batch_graph_label)
                    logits = model(unaug_batch_graph, unaug_batch_size)
                    logits = logits.argmax(1).tolist()
                    if i == 0:
                        indices_record = logits
                        batch_y_record = unaug_batch_label
                    else:
                        indices_record = torch.cat((torch.LongTensor(indices_record), torch.LongTensor(logits)), 0).tolist()
                        batch_y_record = torch.cat((torch.LongTensor(batch_y_record), torch.LongTensor(unaug_batch_label)), 0).tolist()
                acc = accuracy_score(indices_record, batch_y_record)
                f1 = f1_score(indices_record, batch_y_record)
                print(each_epoch, loss_value, acc, f1)
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
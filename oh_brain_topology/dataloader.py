import os.path
import torch
import random
from utils import read_data_from_pickle


def build_dataloader(dataset_config):
    data_dir = dataset_config['data_dir']
    device = dataset_config['device']
    total_epoch = dataset_config['total_epoch']
    aug_method = dataset_config['aug_method']
    train_id_list = dataset_config['train_id_list']
    eval_id_list = dataset_config['eval_id_list']

    train_dataloader = Dataloader(data_dir, total_epoch, aug_method, train_id_list, device)
    eval_dataloader = Dataloader(data_dir, total_epoch, aug_method, eval_id_list, device)
    return train_dataloader, eval_dataloader


class Dataloader():
    def __init__(self, data_dir, total_epoch, aug_method, chosen_index, device='cpu'):
        self.data_dir = data_dir
        self.total_epoch = total_epoch
        self.aug_method = aug_method
        self.device = device
        self.chosen_index = chosen_index

    def get_batch(self, each_epoch, batch_size):
        pkl_file_path = os.path.join(self.data_dir, self.aug_method,  self.aug_method + '_' + str(each_epoch) + '.pkl')
        data_each_epoch = read_data_from_pickle(pkl_file_path)
        chosen_data_each_epoch = []
        for i in range(len(data_each_epoch)):
            if i in self.chosen_index:
                chosen_data_each_epoch.append(data_each_epoch[i])
        if self.device == 'gpu':
            for i in range(len(chosen_data_each_epoch)):
                chosen_data_each_epoch[i] = (chosen_data_each_epoch[i][0].to(torch.device('cuda:0')), chosen_data_each_epoch[i][1])
        batch_data_each_epoch = [chosen_data_each_epoch[i:i + batch_size] for i in range(0, len(chosen_data_each_epoch), batch_size)]
        batch_data_temp = []
        for each_batch_data in batch_data_each_epoch:
            each_batch_graph_list = []
            each_batch_label_list = []
            for each_graph_label in each_batch_data:
                each_batch_graph_list.append(each_graph_label[0])
                each_batch_label_list.append(each_graph_label[1])
            batch_data_temp.append((each_batch_graph_list, each_batch_label_list))
        return batch_data_temp

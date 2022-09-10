import torch
import os, random
from benchmark.utils import read_data_from_pickle

def build_dataloader(dataset_config):
    pkl_filepath = dataset_config['pkl_filepath']
    train_test_ratio = dataset_config['train_test_ratio']
    device = dataset_config['device']
    pkl_dataset = read_data_from_pickle(pkl_filepath)
    train_id_list = random.sample(pkl_dataset.keys(), int(len(pkl_dataset) * train_test_ratio))
    eval_id_list = [each_id for each_id in pkl_dataset.keys() if each_id not in train_id_list]
    train_data_list = [pkl_dataset[each_key] for each_key in train_id_list]
    eval_data_list = [pkl_dataset[each_key] for each_key in eval_id_list]
    return Dataloader(data=train_data_list, device=device), Dataloader(data=eval_data_list, device=device)


class Dataloader():
    def __init__(self, data, device='cpu'):
        self.data = data
        if device == 'gpu':
            for i in range(len(self.data)):
                self.data[i] = (self.data[i][0].to(torch.device('cuda:0')), self.data[i][1])

    def get_batch(self, batch_size):
        batch_data = [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]
        batch_data_temp = []
        for each_batch_data in batch_data:
            each_batch_graph_list = []
            each_batch_label_list = []
            for each_graph_label in each_batch_data:
                each_batch_graph_list.append(each_graph_label[0])
                each_batch_label_list.append(each_graph_label[1])
            batch_data_temp.append((each_batch_graph_list, each_batch_label_list))
        return batch_data_temp
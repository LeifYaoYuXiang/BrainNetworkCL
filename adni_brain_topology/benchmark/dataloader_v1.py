import os, random
import torch
from benchmark.utils import read_data_from_pickle, write_data_to_pickle


def build_data_pkl_file(graph_dir, subject_label_dict, save_filepath):
    pkl_file_dict = {}
    for each_graph_name in os.listdir(graph_dir):
        graph = read_data_from_pickle(os.path.join(graph_dir, each_graph_name))
        label = subject_label_dict[each_graph_name.split('.')[0]]
        pkl_file_dict[each_graph_name.split('.')[0]] = (graph, label)
    write_data_to_pickle(pkl_file_dict, save_filepath)


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


if __name__ == '__main__':
    graph_dir = r'D:\PycharmProjects\adni_brain_topology\benchmark\data\original_data\graph'
    subject_label_dict_filepath = r'D:\PycharmProjects\adni_brain_topology\benchmark\data\original_data\subject_labels_dict.pkl'
    save_filepath = r'D:\PycharmProjects\adni_brain_topology\benchmark\data\dataloader'
    save_filepath = os.path.join(save_filepath, 'dataloader.pkl')
    subject_label_dict = read_data_from_pickle(subject_label_dict_filepath)
    mapping_dict = {
        0: 0,
        4: 1,
        3: 1
    }
    subject_label_dict_remap = {}
    for key, value in subject_label_dict.items():
        if value in mapping_dict.keys():
            subject_label_dict_remap[key] = mapping_dict[value]
    build_data_pkl_file(graph_dir, subject_label_dict_remap, save_filepath)
    print('finish')
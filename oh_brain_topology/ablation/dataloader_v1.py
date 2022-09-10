import os, random
from scipy import sparse
import dgl
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from ablation.utils import write_data_to_pickle, read_data_from_pickle
percent = 0.15


# {
#   id: (each_graph, label)
# }
def data_preprocess(fc_matrix_dir, subject_id_list, save_filepath):
    graph_dict = {}
    index = 0
    for each_subject_id in subject_id_list:
        print(each_subject_id, index, len(subject_id_list))
        each_fc_matrix = np.loadtxt(os.path.join(fc_matrix_dir, each_subject_id, 'no_aug_no_aug', '0.txt'))
        each_dgl_graph = build_dgl_graph(each_fc_matrix)
        label = 0 if 6001 <= int(each_subject_id) <= 6081 else 1
        graph_dict[each_subject_id] = (each_dgl_graph, label)
        index = index + 1
    write_data_to_pickle(graph_dict, save_filepath)


def build_dgl_graph(fc_matrix):
    min_max_scaler = MinMaxScaler()
    scaled_txt_array = min_max_scaler.fit_transform(fc_matrix.T)
    scaled_txt_array = scaled_txt_array.T
    abs_scaled_txt_array = abs(scaled_txt_array)
    baseline = np.quantile(abs_scaled_txt_array, 1 - percent)
    abs_scaled_txt_array[abs_scaled_txt_array < baseline] = 0
    arr_sparse = sparse.csr_matrix(abs_scaled_txt_array)
    graph = dgl.from_scipy(arr_sparse)
    graph.ndata['feat'] = torch.from_numpy(scaled_txt_array)
    return graph


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
        return [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]


# build the graph data
if __name__ == '__main__':
    dir_path = r'F:\OHComparativeLearning\fc_matrix'
    subject_id_list = os.listdir(r'F:\OHComparativeLearning\fc_matrix')
    save_filepath = r'F:\OHComparativeLearning\dataloader.pkl'
    data_preprocess(dir_path, subject_id_list, save_filepath)
    print('data preprocess finish')

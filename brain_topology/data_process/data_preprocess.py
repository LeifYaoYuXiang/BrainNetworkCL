import joblib
from configparser import ConfigParser

from scipy import sparse
import numpy as np
import torch
import dgl
import os, json
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange

config = ConfigParser()
config.read('../parameters.ini', encoding='UTF-8')


# 加载相关的pkl文件
def load_data_loaders_from_pkl(file_path):
    data_loaders = joblib.load(open(file_path, 'rb'))
    return data_loaders


# 6001-6081为control组,label为0
# 其余的为patients组,label为1
# 将二维数组的txt格式的文件加载，并附上相关的label
# label0的数量：64
# label1的数量：122
def load_txt_into_array(filepath, id):
    txt_info = {}
    txt_array = np.loadtxt(filepath)

    # 使用归一化
    min_max_scaler = MinMaxScaler()
    scaled_txt_array = min_max_scaler.fit_transform(txt_array.T)
    txt_array = scaled_txt_array.T

    txt_info['shape'] = txt_array.shape
    txt_info['data'] = txt_array
    txt_info['id'] = id
    if 6001 <= int(id) <= 6081:
        txt_info['label'] = 0
    else:
        txt_info['label'] = 1
    return txt_info


# 加载批次数据
def load_raw_data(raw_data_txt_path_list, raw_data_label_list, percent=0.1):
    graph_list = []
    label_list = []
    id_list = []

    for i in range(len(raw_data_txt_path_list)):
        txt_info = load_txt_into_array(raw_data_txt_path_list[i], raw_data_label_list[i])
        array = txt_info['data']
        abs_array = abs(txt_info['data'])
        # 生成稀疏矩阵
        baseline = np.quantile(abs_array, 1-percent)
        abs_array[abs_array < baseline] = 0
        arr_sparse = sparse.csr_matrix(abs_array)
        # 生成DGL图结构
        graph = dgl.from_scipy(arr_sparse)
        # 生成DGL图的点特征
        min_max_scaler = MinMaxScaler()
        # 按行归一化
        scaled_array = min_max_scaler.fit_transform(array.T)
        array = scaled_array.T
        graph.ndata['feat'] = torch.from_numpy(array)

        graph_list.append(graph)
        label_list.append(txt_info['label'])
        id_list.append(txt_info['id'])

    return graph_list, label_list, id_list


# 生成 DataLoader
def generate_data_in_one_epoch(data, label, id, batch_size):
    batch_graph_in_one_dataloader = []
    graph_number = len(id)
    for i in range(0, graph_number, batch_size):
        start = i
        if i + batch_size < graph_number:
            end = i + batch_size
        else:
            end = graph_number
        graph_temp_list = data[start: end]

        temp_batch_size = end - start
        temp_batch_graph = dgl.batch(graph_temp_list)
        temp_batch_label = label[start: end]
        batch_graph_in_one_dataloader.append({
            'batch_graph': temp_batch_graph,
            'batch_label': temp_batch_label,
            'batch_size': temp_batch_size
        })
    return batch_graph_in_one_dataloader


def gcl_data_preprocess(voxel_to_bold, bold_to_fc, base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size):
    voxel_to_bold_to_fc = voxel_to_bold + '_' + bold_to_fc
    voxel_to_bold_to_fc_path = os.path.join(save_dir, voxel_to_bold_to_fc)
    os.mkdir(voxel_to_bold_to_fc_path)
    for cv_number in range(len(cv_info)):
        voxel_to_bold_to_fc_cv_number_path = os.path.join(voxel_to_bold_to_fc_path, str(cv_number))
        os.mkdir(voxel_to_bold_to_fc_cv_number_path)

        voxel_to_bold_to_fc_cv_number_train_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'train')
        os.mkdir(voxel_to_bold_to_fc_cv_number_train_path)
        voxel_to_bold_to_fc_cv_number_test_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'test')
        os.mkdir(voxel_to_bold_to_fc_cv_number_test_path)

        train_index = cv_info[cv_number]['train_file_id']
        test_index = cv_info[cv_number]['test_file_id']

        for n_epoch in trange(maximum_epoch_number):

            raw_train_data_txt_path_list = []
            raw_train_label_list = []

            raw_test_data_txt_path_list = []
            raw_test_label_list = []

            # 加载train的数据
            for i in range(len(train_index)):
                train_filepath = base_dir + '/' + train_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                raw_train_data_txt_path_list.append(train_filepath)
                raw_train_label_list.append(train_index[i])
            train_data, train_label, train_id = load_raw_data(raw_train_data_txt_path_list, raw_train_label_list, percent=percent)
            train_batch_graph_in_one_dataloader = generate_data_in_one_epoch(train_data, train_label, train_id, batch_size=batch_size)
            # 生成数据集
            train_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_train_path, str(n_epoch)+'.pkl')
            train_pkl = open(train_pkl_path, 'wb')
            joblib.dump(train_batch_graph_in_one_dataloader, train_pkl)

            # 加载test的数据
            for i in range(len(test_index)):
                test_filepath = base_dir + '/' + test_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                raw_test_data_txt_path_list.append(test_filepath)
                raw_test_label_list.append(test_index[i])
            test_data, test_label, test_id = load_raw_data(raw_test_data_txt_path_list, raw_test_label_list, percent=percent)
            test_batch_graph_in_one_dataloader = generate_data_in_one_epoch(test_data, test_label, test_id, batch_size=batch_size)
            # 生成数据集
            test_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_test_path, str(n_epoch)+'.pkl')
            test_pkl = open(test_pkl_path, 'wb')
            joblib.dump(test_batch_graph_in_one_dataloader, test_pkl)


if __name__ == '__main__':
    cv_info_path = config.get('filepath', 'cv_info_path')
    base_dir = config.get('filepath', 'fc_matrix_dir')
    save_dir = config.get('filepath', 'dataloader_dir')

    with open(cv_info_path, 'r') as cv_file:
        cv_info = json.load(cv_file)

    maximum_epoch_number = 200
    percent = 0.1
    batch_size = 8

    print('no_aug_no_aug')
    gcl_data_preprocess('no_aug', 'no_aug', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('no_aug_ratio_sample')
    gcl_data_preprocess('no_aug', 'ratio_sample', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('no_aug_slide_window')
    gcl_data_preprocess('no_aug', 'slide_window', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_no_aug')
    gcl_data_preprocess('aug', 'no_aug', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_ratio_sample')
    gcl_data_preprocess('aug', 'ratio_sample', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_slide_window')
    gcl_data_preprocess('aug', 'slide_window', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)

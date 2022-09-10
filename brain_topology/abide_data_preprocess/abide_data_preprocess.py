import os, json, joblib
import pickle
from random import sample

import pandas as pd
import torch
import dgl
from scipy import sparse
import numpy as np
from configparser import ConfigParser
from tqdm import trange

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from util_encoder import NpEncoder

config = ConfigParser()
config.read('../parameters.ini', encoding='UTF-8')

cv_info_path = config.get('abide_path_2', 'cv_info_path')
voxel_dir_filepath = config.get('abide_path_2', 'voxel_dir')
data_description_filepath = config.get('abide_path_2', 'data_description')
file_id_label_path = config.get('abide_path_2', 'file_id_label_path')

base_dir = config.get('abide_path_2', 'fc_matrix_dir')
save_dir = config.get('abide_path_2', 'dataloader_dir')

# 0: 71
# 1：81
def data_info_cv():
    df = pd.read_csv(data_description_filepath)
    data_description = {}
    experiment_data_description = {}
    data_label_zero_list = []
    data_label_one_list = []

    for row in df.itertuples():
        FILE_ID = getattr(row, 'FILE_ID')
        label = getattr(row, 'DX_GROUP')
        if label == 1:
            label = 0
        else:
            label = 1
        data_description[FILE_ID] = label
    file_name_list = os.listdir(voxel_dir_filepath)

    for each_filename in file_name_list:
        filename = each_filename.split('.')[0]
        if len(filename.split('_')) == 4:
            filename = filename.split('_')[0] + '_' + filename.split('_')[1]
        else:
            filename = filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
        experiment_data_description[each_filename] = data_description[filename]
        if data_description[filename] == 0:
            data_label_zero_list.append(each_filename)
        elif data_description[filename] == 1:
            data_label_one_list.append(each_filename)
    return data_label_zero_list, data_label_one_list


# 生成k-fold的CV
def k_fold_generate_data(k_fold_times,
                         data_label_one_list, data_label_two_list,
                         label_one_number, label_two_number):
    cv_info_json = []
    sampled_data_label_one_list = sample(data_label_one_list, label_one_number)
    sampled_data_label_two_list = sample(data_label_two_list, label_two_number)
    X = np.array(sampled_data_label_one_list + sampled_data_label_two_list)
    y = np.array([1] * label_one_number + [2] * label_two_number)
    skf = StratifiedKFold(n_splits=k_fold_times)
    for train_index, test_index in skf.split(X, y):
        cv_info_json.append({
            'train': X[train_index].tolist(),
            'test': X[test_index].tolist(),
        })
    return cv_info_json


# 生成cv_info
def generate_cv_info():
    data_label_zero_list, data_label_one_list = data_info_cv()

    cv_info = k_fold_generate_data(k_fold_times=5,
                               data_label_one_list=data_label_zero_list,
                               data_label_two_list=data_label_one_list,
                               label_one_number=71, label_two_number=71)
    # 写入
    json_str = json.dumps(cv_info)
    with open(cv_info_path, 'w') as json_file:
        json_file.write(json_str)


# 生成file_id_label
def generate_file_id_label():
    filepath_dict = {}
    df = pd.read_csv(data_description_filepath)

    filepath_list = os.listdir(voxel_dir_filepath)
    for each_filepath in filepath_list:
        if len(each_filepath.split('_')) == 4:
            filename = each_filepath.split('_')[0] + '_' + each_filepath.split('_')[1]
        else:
            filename = each_filepath.split('_')[0] + '_' + each_filepath.split('_')[1] + '_' + each_filepath.split('_')[2]
        file_label = df.loc[df['FILE_ID'] == filename]['DX_GROUP'].values[0]
        filepath_dict[each_filepath] = file_label

    json_str = json.dumps(filepath_dict, cls=NpEncoder)
    with open(file_id_label_path, 'w') as json_file:
        json_file.write(json_str)
    print('FINISHED')


# 加载批次数据
def load_raw_data(raw_data_txt_path_list, percent=0.1):
    graph_list = []
    for i in range(len(raw_data_txt_path_list)):
        txt_array = np.loadtxt(raw_data_txt_path_list[i])
        min_max_scaler = MinMaxScaler()
        scaled_txt_array = min_max_scaler.fit_transform(txt_array.T)
        txt_array = scaled_txt_array.T
        abs_array = abs(txt_array)

        # 生成稀疏矩阵
        baseline = np.quantile(abs_array, 1-percent)
        abs_array[abs_array < baseline] = 0
        arr_sparse = sparse.csr_matrix(abs_array)

        # 生成DGL图结构
        graph = dgl.from_scipy(arr_sparse)
        # 生成DGL图的点特征
        # # 不做归一化
        # array = txt_array
        # # min_max_scaler = MinMaxScaler()
        # 按行归一化
        scaled_array = min_max_scaler.fit_transform(txt_array.T)
        array = scaled_array.T

        # 按列归一化
        # array = min_max_scaler.fit_transform(txt_array)

        graph.ndata['feat'] = torch.from_numpy(array)
        graph_list.append(graph)
    return graph_list


def generate_data_in_one_epoch(data, label, batch_size):
    batch_graph_in_one_dataloader = []
    graph_number = len(label)
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


def abide_data_preprocess(voxel_to_bold, bold_to_fc, base_dir,
                          maximum_epoch_number, save_dir, cv_info, percent, batch_size):
    file_name_list = os.listdir(voxel_dir_filepath)
    with open(file_id_label_path, 'r') as load_f:
        file_id_label_dict = json.load(load_f)

    data_label_one_list = []
    data_label_two_list = []

    for item in file_name_list:
        if file_id_label_dict[item] == 1:
            data_label_one_list.append(item)
        else:
            data_label_two_list.append(item)

    voxel_to_bold_to_fc = voxel_to_bold + '_' + bold_to_fc
    voxel_to_bold_to_fc_path = os.path.join(save_dir, voxel_to_bold_to_fc)
    # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug'
    os.mkdir(voxel_to_bold_to_fc_path)

    for cv_number in range(len(cv_info)):
        voxel_to_bold_to_fc_cv_number_path = os.path.join(voxel_to_bold_to_fc_path, str(cv_number))
        # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug\0'
        os.mkdir(voxel_to_bold_to_fc_cv_number_path)
        voxel_to_bold_to_fc_cv_number_train_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'train')
        voxel_to_bold_to_fc_cv_number_test_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'test')
        # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug\0\train'
        os.mkdir(voxel_to_bold_to_fc_cv_number_train_path)
        # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug\0\test'
        os.mkdir(voxel_to_bold_to_fc_cv_number_test_path)

        train_index = cv_info[cv_number]['train']
        test_index = cv_info[cv_number]['test']

        for n_epoch in trange(maximum_epoch_number):
            raw_train_data_txt_path_list = []
            raw_test_data_txt_path_list = []

            # 加载train的数据
            train_label = []
            for i in range(len(train_index)):
                train_filepath = base_dir + '/' + train_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                if train_index[i] in data_label_one_list:
                    train_label.append(1)
                else:
                    train_label.append(2)
                raw_train_data_txt_path_list.append(train_filepath)
            train_data = load_raw_data(raw_train_data_txt_path_list, percent=percent)
            train_batch_graph_in_one_dataloader = generate_data_in_one_epoch(train_data, train_label, batch_size=batch_size)
            # 生成数据集
            train_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_train_path, str(n_epoch)+'.pkl')
            with open(train_pkl_path, 'wb') as train_pkl:
                joblib.dump(train_batch_graph_in_one_dataloader, train_pkl)

            # 加载test的数据
            test_label = []
            for i in range(len(test_index)):
                test_filepath = base_dir + '/' + test_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                if test_index[i] in data_label_one_list:
                    test_label.append(1)
                else:
                    test_label.append(2)
                raw_test_data_txt_path_list.append(test_filepath)
            test_data = load_raw_data(raw_test_data_txt_path_list, percent=percent)
            test_batch_graph_in_one_dataloader = generate_data_in_one_epoch(test_data, test_label, batch_size=batch_size)
            # 生成数据集
            test_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_test_path, str(n_epoch)+'.pkl')
            with open(test_pkl_path, 'wb') as test_pkl:
                joblib.dump(test_batch_graph_in_one_dataloader, test_pkl)


def main():
    maximum_epoch_number = 200
    percent = 0.1
    batch_size = 10

    # 读取
    with open(cv_info_path, 'r') as load_f:
        cv_info = json.load(load_f)
    print('no_aug_no_aug')
    abide_data_preprocess('no_aug', 'no_aug', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('no_aug_ratio_sample')
    abide_data_preprocess('no_aug', 'ratio_sample', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    # print('no_aug_slide_window')
    # abide_data_preprocess('no_aug', 'slide_window', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    # print('aug_no_aug')
    # abide_data_preprocess('aug', 'no_aug', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    # print('aug_ratio_sample')
    # abide_data_preprocess('aug', 'ratio_sample', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_slide_window')
    abide_data_preprocess('aug', 'slide_window', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)


if __name__ == '__main__':
    main()

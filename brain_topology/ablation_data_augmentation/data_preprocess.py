import random, os, pickle
import numpy as np
import nibabel as nib
from scipy import sparse
import torch
import dgl
from sklearn.preprocessing import MinMaxScaler
from utils import write_data_to_pickle, read_data_from_pickle, send_notification_email


def load_nifti(filepath):
    nifti = nib.load(filepath)
    nifti = np.array(nifti.get_fdata())
    return nifti


def load_pkl(filepath):
    voxel_array = pickle.load(open(filepath, 'rb'))
    return voxel_array


# 利用 增强 或者是 非增强 的方式将voxel矩阵转化为bold矩阵（增强与否取决于roi_ratio的值）
def voxel_to_bold(file_name, original_data, mask, roi_ratio, file_store=False, file_path=''):
    max_value = int(mask.max())
    min_value = int(mask.min())
    bold = []
    for i in range(min_value, max_value):
        temp_mask = mask == i + 1
        roi_number = np.sum(mask == i + 1)
        if 0 < roi_ratio < 1:
            chosen_number = int(roi_number * roi_ratio)
            available_index = np.arange(0, roi_number-1).tolist()
            chosen_index = random.sample(available_index, chosen_number)
            temp_data = original_data[temp_mask]
            temp_data = temp_data[chosen_index]
        elif roi_ratio == 1:
            temp_data = original_data[temp_mask]
        else:
            raise Exception('ROI Ratio的输入格式不正确')
        temp_data = temp_data.mean(0)
        bold.append(temp_data)

    if file_store:
        if file_path == '':
            file_path = os.getcwd()+os.sep+file_name+'-bold.txt'
        else:
            file_path = os.getcwd() + os.sep + file_path + os.sep + file_name + '-bold.txt'
        np.savetxt(file_path, bold, delimiter='\t')
    return np.mat(bold)


# 以slide_window的方式将bold->fc
def bold_to_fc_augmented_slide_window(file_name, bold_array, win_size, file_store=False, file_path=''):
    length = bold_array.shape[1]
    if win_size > length:
        win_size = length
        start_pos = 0
    else:
        # raise Exception('选取的index已经超出过可选的长度')
        start_pos = np.random.randint(length - win_size)
    timestamp_array = np.arange(start_pos, start_pos + win_size)
    chosen_bold_array = bold_array[:, timestamp_array]

    correlation = np.corrcoef(chosen_bold_array)
    correlation[np.isnan(correlation)] = 0
    row, col = np.diag_indices_from(correlation)
    correlation[row, col] = 0

    if file_store:
        if file_path == '':
            file_path = os.getcwd()+os.sep+ file_name +'-FC_slide_window.txt'
        else:
            file_path = os.getcwd() + os.sep + file_path + os.sep + file_name + '-FC_slide_window.txt'
        np.savetxt(file_path, correlation, delimiter='\t')
    return correlation


# 以ratio sample的方式将bold->fc
def bold_to_fc_augmented_ratio_sample(file_name, bold_array, ratio, file_store=False, file_path='', timestamp_array=''):
    length = bold_array.shape[1]
    if ratio >= 1:
        raise Exception('ratio的输入值有误')
    if timestamp_array == '':
        chosen_length = int(length * ratio)
        if chosen_length == 0:
            chosen_length = length
        timestamp_array = np.array(random.sample(range(0, length), chosen_length))
    chosen_bold_array = bold_array[:, timestamp_array]
    correlation = np.corrcoef(chosen_bold_array)
    correlation[np.isnan(correlation)] = 0
    row, col = np.diag_indices_from(correlation)
    correlation[row, col] = 0
    if file_store:
        if file_path == '':
            file_path = os.getcwd()+os.sep + file_name + '-FC_ratio_sample.txt'
        else:
            file_path = os.getcwd() + os.sep + file_path + os.sep + file_name + '-FC_ratio_sample.txt'
        np.savetxt(file_path, correlation, delimiter='\t')
    return correlation, timestamp_array


# 以非增强的方式将bold——>fc
def bold_to_fc(file_name, bold_array, file_store=False, file_path=''):
    correlation = np.corrcoef(bold_array)
    correlation[np.isnan(correlation)] = 0
    row, col = np.diag_indices_from(correlation)
    correlation[row, col] = 0
    if file_store:
        if file_path == '':
            file_path = os.getcwd()+os.sep + file_name + '-FC.txt'
        else:
            file_path = os.getcwd() + os.sep + file_path + os.sep + file_name + '-FC.txt'
        np.savetxt(file_path, correlation, delimiter='\t')
    return correlation


# 数据增强的总开关
def data_augmentation(file_id, voxel_to_bold_type, bold_to_fc_type, voxel_filepath, mask, percent):
    original_data = load_pkl(voxel_filepath)
    if voxel_to_bold_type == 'NoAug':
        bold_matrix = voxel_to_bold(file_name=file_id, original_data=original_data, mask=mask, roi_ratio=1)
    else:
        bold_matrix = voxel_to_bold(file_name=file_id, original_data=original_data, mask=mask, roi_ratio=0.3)

    if bold_to_fc_type == 'NoAug':
        fc_array = bold_to_fc(file_name=file_id, bold_array=bold_matrix)
    elif bold_to_fc_type == 'SlideWindow':
        fc_array = bold_to_fc_augmented_slide_window(file_name=file_id, bold_array=bold_matrix, win_size=80)
    else:
        fc_array, timestamp_array = bold_to_fc_augmented_ratio_sample(file_name=file_id, bold_array=bold_matrix,
                                                                      ratio=0.3)
    fc_array[np.isnan(fc_array)] = 0
    graph = generate_graph(fc_array, percent)
    return graph


# 由txt文件生成 DGL图结构
def generate_graph(txt_array, percent):
    # 使用归一化
    min_max_scaler = MinMaxScaler()
    scaled_txt_array = min_max_scaler.fit_transform(txt_array.T)
    array = scaled_txt_array.T
    abs_array = abs(array)
    # 生成稀疏矩阵
    baseline = np.quantile(abs_array, 1 - percent)
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
    return graph


# 数据处理的总entry
def main(mode, data_preprocess_config, chosen_index, subject_label_dict):
    assert mode in ['preprocess', 'dataloader']
    if mode == 'preprocess':
        voxel_dir = data_preprocess_config['voxel_dir']
        graph_dir = data_preprocess_config['graph_dir']
        augmentation_method = data_preprocess_config['augmentation_method']
        n_epoch = data_preprocess_config['n_epoch']
        mask = load_nifti(data_preprocess_config['mask_file_path'])
        voxel_to_bold_type = augmentation_method.split('_')[0]
        bold_to_fc_type = augmentation_method.split('_')[1]
        percent = data_preprocess_config['percent']
        file_id_list = data_preprocess_config['file_id_list']
        if not os.path.exists(os.path.join(graph_dir, augmentation_method)):
            os.mkdir(os.path.join(graph_dir, augmentation_method))
        for each_epoch in range(n_epoch):
            if not os.path.exists(os.path.join(graph_dir, augmentation_method, str(each_epoch))):
                os.mkdir(os.path.join(graph_dir, augmentation_method, str(each_epoch)))
            # each_voxel_file_list = os.listdir(voxel_dir)
            for i in range(len(file_id_list)):
                each_voxel_file = file_id_list[i] + '.pkl'
                print('AUGMENT GRAPH FINISH: ' + str(i) + ':' + str(len(file_id_list)))
                file_id = each_voxel_file.split('.')[0]
                voxel_filepath = os.path.join(os.path.join(voxel_dir, each_voxel_file))
                augmented_graph = data_augmentation(file_id, voxel_to_bold_type, bold_to_fc_type, voxel_filepath, mask, percent)
                with open(os.path.join(graph_dir, augmentation_method, str(each_epoch), file_id+'.pkl'), 'wb') as pickle_file:
                    pickle.dump(augmented_graph, pickle_file)
            print('EPOCH FINISH: ' + str(each_epoch) + ':' + str(n_epoch))
    else:
        graph_dir = data_preprocess_config['graph_dir']
        dataloader_dir = data_preprocess_config['dataloader_dir']
        augmentation_method = data_preprocess_config['augmentation_method']
        n_epoch = data_preprocess_config['n_epoch']
        batch_size = data_preprocess_config['batch_size']
        if not os.path.exists(os.path.join(dataloader_dir, augmentation_method)):
            os.mkdir(os.path.join(dataloader_dir, augmentation_method))
        for cv_index in range(data_preprocess_config['cv']):
            print(str(cv_index) + 'split start')
            # each_train_cv_list = []
            # each_test_cv_list = []
            train_index = chosen_index[cv_index]['train']
            test_index = chosen_index[cv_index]['test']
            # random.shuffle(train_index)
            # random.shuffle(test_index)
            train_index = utils_cut_list(train_index, batch_size)
            test_index = utils_cut_list(test_index, batch_size)

            for each_epoch in range(n_epoch):
                print(str(each_epoch) + 'epoch generation start')
                each_train_epoch_list = []
                each_test_epoch_list = []

                for each_batch in train_index:
                    each_train_batch_list = []
                    for each_graph in each_batch:
                        each_label = subject_label_dict[each_graph]
                        each_dgl_graph = load_pkl(os.path.join(graph_dir, augmentation_method, str(each_epoch), each_graph+'.pkl'))
                        each_train_batch_list.append((each_dgl_graph, each_label))
                    each_train_epoch_list.append(each_train_batch_list)

                for each_batch in test_index:
                    each_test_batch_list = []
                    for each_graph in each_batch:
                        each_label = subject_label_dict[each_graph]
                        each_dgl_graph = load_pkl(
                            os.path.join(graph_dir, augmentation_method, str(each_epoch), each_graph + '.pkl'))
                        each_test_batch_list.append((each_dgl_graph, each_label))
                    each_test_epoch_list.append(each_test_batch_list)

                # each_train_cv_list.append(each_train_epoch_list)
                # each_test_cv_list.append(each_test_epoch_list)
                with open(os.path.join(dataloader_dir, augmentation_method, str(cv_index)+'_'+str(each_epoch) + '_train.pkl'), 'wb') as pickle_file:  # 将数据写入pkl文件
                    pickle.dump(each_train_epoch_list, pickle_file)
                with open(os.path.join(dataloader_dir, augmentation_method, str(cv_index)+'_'+str(each_epoch) + '_test.pkl'), 'wb') as pickle_file:  # 将数据写入pkl文件
                    pickle.dump(each_test_epoch_list, pickle_file)
                # print(str(each_epoch) + 'epoch generation finished')
            # pickle 保存
            # each_split_train_test_data = {
            #     'train': each_train_cv_list,
            #     'test': each_test_cv_list
            # }
            # with open(os.path.join(dataloader_dir, augmentation_method, str(cv_index)+'.pkl'), 'wb') as pickle_file:  # 将数据写入pkl文件
            #     pickle.dump(each_split_train_test_data, pickle_file)
            print(str(cv_index) + 'split finish')


# 按找固定长度切分数组
def utils_cut_list(lists, cut_len):
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)
    return res_data


# 根据 file_id list 生成文件
def generate_chosen_index(subject_label_dict, split_time, train_test_split_ratio):
    # 根据不同的label标签获得我们的id list
    label_0_file_id_list = []
    label_1_file_id_list = []
    label_2_file_id_list = []
    for k, v in subject_label_dict.items():
        if v == 0: label_0_file_id_list.append(k)
        if v == 1: label_1_file_id_list.append(k)
        if v == 2: label_2_file_id_list.append(k)
    label_0_file_id_list = random.sample(label_0_file_id_list, 60)
    label_1_file_id_list = random.sample(label_1_file_id_list, 60)
    file_id_list = label_0_file_id_list + label_1_file_id_list + label_2_file_id_list

    chosen_index = {}
    for i in range(split_time):
        random.shuffle(file_id_list)
        split_index = int(len(file_id_list) * train_test_split_ratio)
        chosen_index[i] = {
            'train': file_id_list[0: split_index],
            'test': file_id_list[split_index: len(file_id_list)],
        }
    return chosen_index, file_id_list



"""
保存的数据类型：

生成数据的格式如下：
[
    [epoch_1 [batch_1 (graph_1, label_1), (graph_2, label_2), (graph_3, label_3)], [batch_2]],
    [epoch_2 [batch_1 (graph_1, label_1), (graph_2, label_2), (graph_3, label_3)], [batch_2]]
]
"""


if __name__ == '__main__':
    random.seed(1024)
    import warnings
    warnings.filterwarnings('ignore')
    # 对数据进行预处理
    subject_label_dict = load_pkl('/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/subject_labels_dict.pkl')
    mapping_dict = {
        0: 0,
        1: 1,
        2: 1,
        4: 1,
        3: 2,
    }
    for key, value in subject_label_dict.items():
        subject_label_dict[key] = mapping_dict[value]
    # chosen_index, file_id_list = generate_chosen_index(subject_label_dict, split_time=5, train_test_split_ratio=0.7)
    # write_data_to_pickle(chosen_index, filepath='/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/chosen_index.pkl')
    # write_data_to_pickle(file_id_list, filepath='/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/file_id_list.pkl')

    chosen_index = read_data_from_pickle('/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/chosen_index.pkl')
    file_id_list = read_data_from_pickle('/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/file_id_list.pkl')

    mode = 'dataloader'
    data_preprocess_config = {
        'voxel_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/voxel',
        'graph_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/graph',
        'dataloader_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/dataloader',
        'mask_file_path': '/home/xuesong/yuxiang_yao/brain_topology_data/OHComparativeLearning/BN_Atlas_246_3mm.nii',
        'n_epoch': 200,
        'percent': 0.15,
        'file_id_list': file_id_list,
        'cv': 5,
        'batch_size': 8,
    }
    print('NoAug_NoAug Start')
    data_preprocess_config['augmentation_method'] = 'NoAug_NoAug'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict)
    print('NoAug_NoAug Finish')

    print('NoAug_RatioSample Start')
    data_preprocess_config['augmentation_method'] = 'NoAug_RatioSample'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict)
    print('NoAug_RatioSample Finish')

    print('NoAug_SlideWindow Start')
    data_preprocess_config['augmentation_method'] = 'NoAug_SlideWindow'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict)
    print('NoAug_SlideWindow Finish')

    print('Aug_NoAug Start')
    data_preprocess_config['augmentation_method'] = 'Aug_NoAug'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict)
    print('Aug_NoAug Finish')

    print('Aug_RatioSample Start')
    data_preprocess_config['augmentation_method'] = 'Aug_RatioSample'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict)
    print('Aug_RatioSample Finish')

    print('Aug_SlideWindow Start')
    data_preprocess_config['augmentation_method'] = 'Aug_SlideWindow'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict)
    print('Aug_SlideWindow Finish')

    # send_notification_email('batch split finish')

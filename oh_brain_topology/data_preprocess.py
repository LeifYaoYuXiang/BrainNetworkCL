import os, random, re
from scipy import sparse
import dgl
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from utils import write_data_to_pickle, read_data_from_pickle, load_nifti, send_notification_email

percent = 0.15
mask_file_path = r'F:\OHComparativeLearning\BN_Atlas_246_3mm.nii'
mask = load_nifti(mask_file_path)


def load_voxel_file(voxel_file_dir, subject_id_list):
    voxel_data_label_list = []
    for each_voxel_file in os.listdir(voxel_file_dir):
        each_subject_id = int(re.findall(r'\d+', each_voxel_file.split('.')[0])[0])
        if each_subject_id in subject_id_list:
            voxel_data = read_data_from_pickle(os.path.join(voxel_file_dir, each_voxel_file.split('.')[0]+'.pkl'))
            each_subject_id = int(re.findall(r'\d+', each_voxel_file.split('.')[0])[0])
            label_data = 0 if 6001 <= int(each_subject_id) <= 6081 else 1
            voxel_data_label_list.append((voxel_data, label_data))
    return voxel_data_label_list


def phase_1_no_aug(each_voxel_data):
    max_value = int(mask.max())
    min_value = int(mask.min())
    each_bold_data = []
    for i in range(min_value, max_value):
        temp_mask = mask == i + 1
        temp_data = each_voxel_data[temp_mask]
        temp_data = temp_data.mean(0)
        each_bold_data.append(temp_data)
    return np.mat(each_bold_data)


def phase_1_voxel_sampling(each_voxel_data, roi_ratio=0.3):
    max_value = int(mask.max())
    min_value = int(mask.min())
    each_bold_data = []
    for i in range(min_value, max_value):
        temp_mask = mask == i + 1
        roi_number = np.sum(mask == i + 1)
        chosen_number = int(roi_number * roi_ratio)
        available_index = np.arange(0, roi_number-1).tolist()
        chosen_index = random.sample(available_index, chosen_number)
        temp_data = each_voxel_data[temp_mask]
        temp_data = temp_data[chosen_index]
        temp_data = temp_data.mean(0)
        each_bold_data.append(temp_data)
    return np.mat(each_bold_data)


def phase_1_voxel_radial_sampling(each_voxel_data, roi_ratio=0.3):
    max_value = int(mask.max())
    min_value = int(mask.min())
    each_bold_data = []
    for i in range(min_value, max_value):
        temp_mask = mask == i + 1
        roi_index = np.argwhere(temp_mask == True)
        center_point_axis = np.argwhere(temp_mask == True).mean(0).astype(int)
        dist_diff = abs(roi_index - center_point_axis).sum(1)
        chosen_index = np.argwhere(dist_diff < dist_diff.max() / 2).reshape(-1)
        temp_data = each_voxel_data[temp_mask]
        temp_data = temp_data[chosen_index]
        temp_data = temp_data.mean(0)
        each_bold_data.append(temp_data)
    return np.mat(each_bold_data)


def phase_2_no_aug(each_bold_data):
    each_fc_data = np.corrcoef(each_bold_data)
    each_fc_data[np.isnan(each_fc_data)] = 0
    row, col = np.diag_indices_from(each_fc_data)
    each_fc_data[row, col] = 0
    return each_fc_data


def phase_2_slide_window(each_bold_data, win_size=80):
    length = each_bold_data.shape[1]
    if win_size > length:
        raise Exception('选取的index已经超出过可选的长度')
    start_pos = np.random.randint(length - win_size)
    timestamp_array = np.arange(start_pos, start_pos + win_size)
    chosen_bold_array = each_bold_data[:, timestamp_array]
    each_fc_data = np.corrcoef(chosen_bold_array)
    each_fc_data[np.isnan(each_fc_data)] = 0
    row, col = np.diag_indices_from(each_fc_data)
    each_fc_data[row, col] = 0
    return each_fc_data


def phase_2_ratio_sample(each_bold_data, ratio=0.3, timestamp_array=''):
    length = each_bold_data.shape[1]
    if ratio >= 1:
        raise Exception('ratio的输入值有误')
    if timestamp_array == '':
        chosen_length = int(length * ratio)
        timestamp_array = np.array(random.sample(range(0, length), chosen_length))
    chosen_bold_array = each_bold_data[:, timestamp_array]
    each_fc_data = np.corrcoef(chosen_bold_array)
    each_fc_data[np.isnan(each_fc_data)] = 0
    row, col = np.diag_indices_from(each_fc_data)
    each_fc_data[row, col] = 0
    return each_fc_data


phase_1_aug_method_dict = {
    'phase_1_no_aug': phase_1_no_aug,
    'phase_1_voxel_sampling': phase_1_voxel_sampling,
    'phase_1_voxel_radial_sampling': phase_1_voxel_radial_sampling,
}
phase_2_aug_method_dict = {
    'phase_2_no_aug': phase_2_no_aug,
    'phase_2_slide_window': phase_2_slide_window,
    'phase_2_ratio_sample': phase_2_ratio_sample,
}


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


def data_augmentation_one_epoch(voxel_data_label_list, phase_1_aug_method, phase_2_aug_method):
    fc_data_label_list = []
    for each_voxel_data_label in voxel_data_label_list:
        each_voxel_data = each_voxel_data_label[0]
        each_label_data = each_voxel_data_label[1]
        each_bold_data = phase_1_aug_method_dict[phase_1_aug_method](each_voxel_data)
        each_fc_data = phase_2_aug_method_dict[phase_2_aug_method](each_bold_data)
        fc_data_label_list.append((each_fc_data, each_label_data))
    return fc_data_label_list


def data_augmentation(voxel_data_label_list, phase_1_aug_method, phase_2_aug_method, n_epoch, save_dir):
    for each_epoch in range(n_epoch):
        print(each_epoch, n_epoch)
        graph_data_label_list = []
        fc_data_label_list = data_augmentation_one_epoch(voxel_data_label_list, phase_1_aug_method, phase_2_aug_method)
        for each_fc_data_label in fc_data_label_list:
            each_graph_data = build_dgl_graph(each_fc_data_label[0])
            graph_data_label_list.append((each_graph_data, each_fc_data_label[1]))
        write_data_to_pickle(graph_data_label_list, os.path.join(save_dir, phase_1_aug_method + '_' + phase_2_aug_method + '_' + str(each_epoch)+ '.pkl'))


if __name__ == '__main__':
    voxel_file_dir = r'F:\OHComparativeLearning\voxel_signals'
    subject_filename = os.listdir(r'F:\OHComparativeLearning\fc_matrix')
    subject_id_list = [int(x.split('.')[0]) for x in subject_filename]
    # subject_id_list = [6001]
    n_epoch = 200
    save_dir = r'F:\OHComparativeLearning\graph_list_new'
    print(subject_id_list)
    voxel_data_label_list = load_voxel_file(voxel_file_dir, subject_id_list)
    print('voxel file load finish')
    available_aug_method = [# ('phase_1_no_aug', 'phase_2_no_aug'),
                            # ('phase_1_no_aug', 'phase_2_slide_window'),
                            ('phase_1_no_aug', 'phase_2_ratio_sample'),
                            ('phase_1_voxel_sampling', 'phase_2_no_aug'),
                            ('phase_1_voxel_sampling', 'phase_2_slide_window'),
                            ('phase_1_voxel_sampling', 'phase_2_ratio_sample'),
                            ('phase_1_voxel_radial_sampling', 'phase_2_no_aug'),
                            ('phase_1_voxel_radial_sampling', 'phase_2_slide_window'),
                            ('phase_1_voxel_radial_sampling', 'phase_2_ratio_sample'),
                            ]
    for phase_1_2_aug_method in available_aug_method:
        phase_1_aug_method = phase_1_2_aug_method[0]
        phase_2_aug_method = phase_1_2_aug_method[1]
        print(phase_1_aug_method, phase_2_aug_method)
        os.mkdir(os.path.join(save_dir, phase_1_aug_method + '_' + phase_2_aug_method))
        data_augmentation(voxel_data_label_list, phase_1_aug_method, phase_2_aug_method, n_epoch, os.path.join(save_dir, phase_1_aug_method + '_' + phase_2_aug_method))
        # send_notification_email(phase_1_aug_method + '-' + phase_2_aug_method)




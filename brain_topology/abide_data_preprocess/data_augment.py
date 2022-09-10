import os, random, re, pickle
from configparser import ConfigParser

import numpy as np
import nibabel as nib
from tqdm import trange

# config = ConfigParser()
# config.read('../parameters.ini', encoding='UTF-8')


# 将相关的nii文件加载进来
def load_nifti(filepath):
    nifti = nib.load(filepath)
    nifti = np.array(nifti.get_fdata())
    return nifti


# 将相关的pkl文件加载进来
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
        raise Exception('选取的index已经超出过可选的长度')
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
def bold_to_fc_augmented_ratio_sample(file_name, bold_array, ratio,
                                      file_store=False, file_path='', timestamp_array=''):
    length = bold_array.shape[1]
    if ratio >= 1:
        raise Exception('ratio的输入值有误')
    if timestamp_array == '':
        chosen_length = int(length * ratio)
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


def data_augmentation_process(file_id, voxel_to_bold_type, bold_to_fc_type, voxel_filepath, fc_save_dir, mask_file_path, augmented_times):
    mask = load_nifti(mask_file_path)
    original_data = load_pkl(voxel_filepath)
    for i in range(len(voxel_to_bold_type)):
        for j in range(len(bold_to_fc_type)):
            # 判断相关文件夹是否存在，如果不存在就创建
            if not os.path.exists(os.path.join(fc_save_dir, file_id)):
                os.mkdir(os.path.join(fc_save_dir, file_id))
            augmentation_type = str(voxel_to_bold_type[i]) + '_' + str(bold_to_fc_type[j])
            fc_save_dir_main = os.path.join(fc_save_dir, file_id, augmentation_type)
            if not os.path.exists(fc_save_dir_main):
                os.mkdir(fc_save_dir_main)

            if voxel_to_bold_type[i] == 'no_aug':
                bold_matrix = voxel_to_bold(file_name=file_id, original_data=original_data, mask=mask, roi_ratio=1)
            else:
                bold_matrix = voxel_to_bold(file_name=file_id, original_data=original_data, mask=mask, roi_ratio=0.3)

            if bold_to_fc_type[j] == 'no_aug':
                fc_array = bold_to_fc(file_name=file_id, bold_array=bold_matrix)
            elif bold_to_fc_type[j] == 'slide_window':
                fc_array = bold_to_fc_augmented_slide_window(file_name=file_id, bold_array=bold_matrix, win_size=80)
            else:
                fc_array, timestamp_array = bold_to_fc_augmented_ratio_sample(file_name=file_id, bold_array=bold_matrix,ratio=0.3)
            fc_array[np.isnan(fc_array)] = 0
            np.savetxt(os.path.join(fc_save_dir_main, str(augmented_times)+'.txt'), fc_array, delimiter='\t')


# def main():
#     augmented_times = 200
#
#     voxel_dir_path = config.get('filepath', 'voxel_dir')
#     fc_saved_dir_path = config.get('filepath', 'fc_matrix_dir')
#     best_id_file_path = config.get('filepath', 'best_id_path')
#     mask_file_path = config.get('filepath', 'mask_file_path')
#
#     # 加载需要进行数据增强的文件id
#     best_id_list = []
#     f = open(best_id_file_path)
#     for line in f:
#         best_id_list = best_id_list + re.findall(r"\d+\.?\d*", str(line))
#     augment_file_number = len(best_id_list)
#
#     voxel_file_augment_path_list = []
#     file_id_augment_list = []
#     label_augment_list = []
#
#     voxel_to_bold_type = ['no_aug', 'aug']
#     bold_to_fc_type = ['no_aug', 'ratio_sample', 'slide_window']
#
#     # 检查每一个voxel文件，判断是否需要进行数据增强处理
#     voxel_files_dir = os.listdir(voxel_dir_path)
#     for voxel_file in voxel_files_dir:
#         file_id = re.findall(r"\d+\.?\d*", voxel_file.split('.')[0])[0]
#         if file_id in best_id_list:
#             voxel_file_augment_path_list.append(voxel_dir_path + '/' + voxel_file)
#             file_id_augment_list.append(file_id)
#             if 6001 <= int(file_id) <= 6081:
#                 label_augment_list.append(0)
#             else:
#                 label_augment_list.append(1)
#     # 开始增强数据
#     for i in trange(augmented_times):
#         for j in range(augment_file_number):
#             data_augmentation_process(file_id=file_id_augment_list[j],
#                                       voxel_to_bold_type=voxel_to_bold_type,
#                                       bold_to_fc_type=bold_to_fc_type,
#                                       voxel_filepath=voxel_file_augment_path_list[j],
#                                       fc_save_dir=fc_saved_dir_path,
#                                       mask_file_path=mask_file_path,
#                                       augmented_times=i)
#             print('第'+str(j+1)+'个数据处理完成：' + str(file_id_augment_list[j]))

#
# if __name__ == '__main__':
#     main()
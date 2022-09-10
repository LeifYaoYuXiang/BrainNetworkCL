import os
from configparser import ConfigParser
import numpy as np
from tqdm import trange

from data_augment import load_nifti, voxel_to_bold, bold_to_fc_augmented_slide_window, bold_to_fc, \
    bold_to_fc_augmented_ratio_sample

config = ConfigParser()
config.read('../parameters.ini', encoding='UTF-8')


def data_augmentation_process(file_id, voxel_to_bold_type, bold_to_fc_type, voxel_filepath, fc_save_dir, mask_file_path, augmented_times):
    mask = load_nifti(mask_file_path)
    original_data = load_nifti(voxel_filepath)
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
                bold_matrix = voxel_to_bold(file_name=file_id, original_data=original_data, mask=mask, roi_ratio=0.5)
            if bold_to_fc_type[j] == 'no_aug':
                fc_array = bold_to_fc(file_name=file_id, bold_array=bold_matrix)
            elif bold_to_fc_type[j] == 'slide_window':
                fc_array = bold_to_fc_augmented_slide_window(file_name=file_id, bold_array=bold_matrix, win_size=100)
            else:
                fc_array, timestamp_array = bold_to_fc_augmented_ratio_sample(file_name=file_id, bold_array=bold_matrix, ratio=0.5)
            fc_array[np.isnan(fc_array)] = 0
            np.savetxt(os.path.join(fc_save_dir_main, str(augmented_times)+'.txt'), fc_array, delimiter='\t')


def augmentation_main(file_id_list):
    augmented_times = 200
    voxel_dir_path = config.get('abide_path', 'voxel_dir')
    fc_saved_dir_path = config.get('abide_path', 'fc_matrix_dir')
    mask_file_path = config.get('filepath', 'mask_file_path')

    augment_file_number = len(file_id_list)
    print(augment_file_number)

    voxel_file_augment_path_list = []

    voxel_to_bold_type = ['no_aug', 'aug']
    bold_to_fc_type = ['no_aug', 'ratio_sample', 'slide_window']
    # 检查每一个voxel文件，判断是否需要进行数据增强处理
    voxel_files_dir = os.listdir(voxel_dir_path)
    for voxel_file in voxel_files_dir:
        if voxel_file in file_id_list:
            voxel_file_augment_path_list.append(voxel_dir_path + '/' + voxel_file)

    # 开始增强数据
    for i in trange(augmented_times):
        for j in range(augment_file_number):
            data_augmentation_process(file_id=file_id_list[j],
                                      voxel_to_bold_type=voxel_to_bold_type,
                                      bold_to_fc_type=bold_to_fc_type,
                                      voxel_filepath=voxel_file_augment_path_list[j],
                                      fc_save_dir=fc_saved_dir_path,
                                      mask_file_path=mask_file_path,
                                      augmented_times=i)
            print('第'+str(j+1)+'个数据处理完成：' + str(file_id_list[j]))


if __name__ == '__main__':
    voxel_dir = config.get('abide_path', 'voxel_dir')
    file_id_list = os.listdir(voxel_dir)
    augmentation_main(file_id_list=file_id_list)

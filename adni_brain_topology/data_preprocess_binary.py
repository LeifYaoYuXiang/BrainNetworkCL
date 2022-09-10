# 为二分类问题做数据处理
import os, random
from data_preprocess import load_pkl, main
from utils import write_data_to_pickle, read_data_from_pickle, send_notification_email


# 根据 file_id list 生成文件
def generate_chosen_index(subject_label_dict, split_time, train_test_split_ratio, voxel_dir):
    # 根据不同的label标签获得我们的id list
    label_0_file_id_list = []
    label_1_file_id_list = []
    all_voxel_file = os.listdir(voxel_dir)
    all_voxel_file = [i.split('.')[0] for i in all_voxel_file]
    for k, v in subject_label_dict.items():
        if v == 0 and k in all_voxel_file: label_0_file_id_list.append(k)
        if v == 1 and k in all_voxel_file: label_1_file_id_list.append(k)

    # for label 0 and label 1
    label_0_file_id_list = random.sample(label_0_file_id_list, 90)
    label_1_file_id_list = random.sample(label_1_file_id_list, 90)

    file_id_list = label_0_file_id_list + label_1_file_id_list

    chosen_index = {}
    for i in range(split_time):
        random.shuffle(file_id_list)
        split_index = int(len(file_id_list) * train_test_split_ratio)
        chosen_index[i] = {
            'train': file_id_list[0: split_index],
            'test': file_id_list[split_index: len(file_id_list)],
        }
    return chosen_index, file_id_list


if __name__ == '__main__':
    random.seed(1024)
    import warnings

    warnings.filterwarnings('ignore')

    subject_label_dict = load_pkl(
        '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/subject_labels_dict.pkl')
    voxel_dir = '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/voxel'

    mapping_dict = {
        0: 0,
        4: 1,
        3: 1
    }

    subject_label_dict_remap = {}
    for key, value in subject_label_dict.items():
        if value in mapping_dict.keys():
            subject_label_dict_remap[key] = mapping_dict[value]

    # chosen_index, file_id_list = generate_chosen_index(subject_label_dict_remap, split_time=5,
    #                                                    train_test_split_ratio=0.7, voxel_dir=voxel_dir)
    # write_data_to_pickle(chosen_index,
    #                      filepath='/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/chosen_index_binary.pkl')
    # write_data_to_pickle(file_id_list,
    #                      filepath='/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/file_id_list_binary.pkl')
    # print('split finish')

    chosen_index = read_data_from_pickle(
        '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/chosen_index_binary.pkl')
    file_id_list = read_data_from_pickle(
        '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/file_id_list_binary.pkl')

    mode = 'dataloader'
    data_preprocess_config = {
        'voxel_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/voxel',
        'graph_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/graph_binary',
        'dataloader_dir': '/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/dataloader_binary',
        'mask_file_path': '/home/xuesong/yuxiang_yao/brain_topology_data/OHComparativeLearning/BN_Atlas_246_3mm.nii',
        'n_epoch': 200,
        'percent': 0.15,
        'file_id_list': file_id_list,
        'cv': 5,
        'batch_size': 8,
    }
    print('NoAug_NoAug Start')
    data_preprocess_config['augmentation_method'] = 'NoAug_NoAug'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict_remap)
    print('NoAug_NoAug Finish')

    print('NoAug_RatioSample Start')
    data_preprocess_config['augmentation_method'] = 'NoAug_RatioSample'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict_remap)
    print('NoAug_RatioSample Finish')

    print('NoAug_SlideWindow Start')
    data_preprocess_config['augmentation_method'] = 'NoAug_SlideWindow'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict_remap)
    print('NoAug_SlideWindow Finish')

    print('Aug_NoAug Start')
    data_preprocess_config['augmentation_method'] = 'Aug_NoAug'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict_remap)
    print('Aug_NoAug Finish')

    print('Aug_RatioSample Start')
    data_preprocess_config['augmentation_method'] = 'Aug_RatioSample'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict_remap)
    print('Aug_RatioSample Finish')

    print('Aug_SlideWindow Start')
    data_preprocess_config['augmentation_method'] = 'Aug_SlideWindow'
    main(mode, data_preprocess_config, chosen_index, subject_label_dict_remap)
    print('Aug_SlideWindow Finish')

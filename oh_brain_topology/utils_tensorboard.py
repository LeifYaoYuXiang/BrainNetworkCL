import os
import configparser

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import seaborn as sns
ini_file = 'configuration.ini'


def findMaxAverage(nums, k: int):
    low = 0
    all = sum(nums[0:k])
    avg = all
    for i in range(k, len(nums)):
        all = all + nums[i] - nums[low]
        low += 1
        avg = max(avg, all)
    return avg / k

naming_dict = {
    'phase_1_no_aug_phase_2_slide_window': 'AugApproach1',
    'phase_1_no_aug_phase_2_ratio_sample': 'AugApproach2',
    'phase_1_voxel_sampling_phase_2_no_aug': 'AugApproach3',
    'phase_1_voxel_sampling_phase_2_slide_window': 'AugApproach4',
    'phase_1_voxel_sampling_phase_2_ratio_sample': 'AugApproach5',
    'phase_1_voxel_radial_sampling_phase_2_no_aug': 'AugApproach6',
    'phase_1_voxel_radial_sampling_phase_2_slide_window': 'AugApproach7',
    'phase_1_voxel_radial_sampling_phase_2_ratio_sample': 'AugApproach8',
    'phase_1_no_aug_phase_2_no_aug': 'AugApproach9',
}

def tensorboard_reading(run_dir):
    k = 10
    index = 1
    exp_dir = os.listdir(run_dir)
    acc_exp_result = {}
    f1_exp_result = {}
    aug_id_dict = {}
    for each_exp_dir in exp_dir:
        # 读取 Configuration
        parser = configparser.ConfigParser()
        parser.read(os.path.join(run_dir, each_exp_dir, ini_file))
        aug_1_method = parser.get('DATASET', 'aug_1_method')
        aug_2_method = parser.get('DATASET', 'aug_2_method')
        # if aug_1_method == 'phase_1_no_aug_phase_2_no_aug' and aug_2_method == 'phase_1_no_aug_phase_2_no_aug':
        #     break
        if aug_1_method not in aug_id_dict.keys():
            aug_id_dict[aug_1_method] = naming_dict[aug_1_method]
            index = index + 1
        if aug_2_method not in aug_id_dict.keys():
            aug_id_dict[aug_2_method] = naming_dict[aug_2_method]
            index = index + 1
        # 读取 Tensorboard 数据
        ea = event_accumulator.EventAccumulator(os.path.join(run_dir, each_exp_dir, os.listdir(os.path.join(run_dir, each_exp_dir))[-1]))
        ea.Reload()
        acc = ea.scalars.Items('Acc')
        f1 = ea.scalars.Items('F1')
        if (aug_1_method, aug_2_method) not in acc_exp_result.keys() and (aug_2_method, aug_1_method) not in acc_exp_result.keys():
            acc_exp_result[(aug_1_method, aug_2_method)] = findMaxAverage([i.value for i in acc], k)
            f1_exp_result[(aug_1_method, aug_2_method)] = findMaxAverage([i.value for i in f1], k)
    # acc_exp_result = sorted(acc_exp_result.items(), key=lambda kv:(kv[1], kv[0]))
    # f1_exp_result = sorted(f1_exp_result.items(), key=lambda kv:(kv[1], kv[0]))
    return acc_exp_result, f1_exp_result, aug_id_dict


def draw_heatmap(aug_id_dict, result, title):
    # build dataframe
    aug_list = aug_id_dict.values()
    df = pd.DataFrame(columns=aug_list, index=aug_list, data=0)
    for k, v in result.items():
        index = aug_id_dict[k[0]]
        column = aug_id_dict[k[1]]
        df.loc[index, column] = v
        df.loc[column, index] = v
    df = df.drop('AugApproach9', axis=0)
    df = df.drop('AugApproach9', axis=1)
    # print(df)
    # draw heatmap
    df = df[['AugApproach1', 'AugApproach2', 'AugApproach3', 'AugApproach4', 'AugApproach5', 'AugApproach6', 'AugApproach7', 'AugApproach8']]
    df = df.sort_index()
    plt.figure(figsize=(10.5, 10.5))
    sns.heatmap(data=df, vmin=0.5, vmax=0.8,
                annot=True, fmt=".4f",
                cmap=sns.diverging_palette(10, 220, sep=80, n=7),
                annot_kws={'size': 8, 'weight': 'normal'}
                )
    # plt.title(title)
    plt.show()


if __name__ == '__main__':
    run_dir = r'D:\PycharmProjects\oh_brain_topology\run'
    acc_exp_result, f1_exp_result, aug_id_dict = tensorboard_reading(run_dir)
    print(aug_id_dict)
    print()
    print(acc_exp_result)
    print()
    print(f1_exp_result)
    draw_heatmap(aug_id_dict, acc_exp_result, 'Heatmap of Augmentation Effect on Acc Metric')
    draw_heatmap(aug_id_dict, f1_exp_result, 'Heatmap of Augmentation Effect on F1 Metric')

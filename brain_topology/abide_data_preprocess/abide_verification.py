import os
import numpy as np
import pandas as pd
import itertools
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from data_process.data_augment import load_nifti, voxel_to_bold, bold_to_fc

collected_data_dir = 'F:\\AbideData_Download_Pipeline\\collected_data'
data_description_filepath = 'F:\\AbideData\\data_description.csv'
facility = ['CALTECH', 'CMU', 'KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAXMUN', 'NYU', 'OHSU', 'OLIN',
            'PITT', 'SBL', 'SDSU', 'STANFORD', 'TRINITY', 'UCLA_1', 'UCLA_2', 'UM_1', 'UM_2',
            'USM', 'YALE']


def select_facility():
    selected_facility_list = []
    optional_facility_number = len(facility)
    for i in range(optional_facility_number):
        optional_facility = list(itertools.combinations(facility, i))
        for j in range(len(optional_facility)):
            selected_facility = list(optional_facility[j])
            selected_facility_list.append(selected_facility)
    return selected_facility_list


def get_available_data(selected_facility_list, df, target_number):
    available_data_dict = dict()
    i = 1
    for selected_facility in selected_facility_list:
        if len(selected_facility) != 0:
            data_path_and_label = dict()

            file_number = 0
            for each_facility in selected_facility:
                file_number = file_number + len(os.listdir(os.path.join(collected_data_dir, each_facility)))

            if file_number >= target_number:
                for each_facility in selected_facility:
                    each_facility = os.path.join(collected_data_dir, each_facility)
                    for all_available_filepath in os.listdir(each_facility):
                        if len(all_available_filepath.split('_')) == 4:
                            file_id = all_available_filepath.split('_')[0]+'_'+all_available_filepath.split('_')[1]
                        else:
                            file_id = all_available_filepath.split('_')[0]+'_'+all_available_filepath.split('_')[1]+'_'+all_available_filepath.split('_')[2]
                        if int(df[df['FILE_ID'] == file_id].values[0][7]) == 1:
                            data_path_and_label[file_id] = 0
                        else:
                            data_path_and_label[file_id] = 1
                print("-".join(selected_facility))
                available_data_dict["-".join(selected_facility)] = data_path_and_label
        print(str(i) + '/' + str(len(selected_facility_list)))
        i = i + 1
    return available_data_dict


def triu_matrix_generation(df, output_path):
    triu_matrix_dict = dict()
    for row in df.itertuples():
        file_id = getattr(row, 'FILE_ID')
        file_path = os.path.join(output_path, file_id+'.txt')
        if os.path.exists(file_path):
            print(file_path)
            fc_matrix = np.loadtxt(file_path)
            fc_matrix_length = fc_matrix.shape[0]
            iu = np.triu_indices(fc_matrix_length)
            triu_matrix = fc_matrix[iu]
            triu_matrix_dict[file_id] = triu_matrix
    return triu_matrix_dict


def judge_fc_matrix(triu_matrix_dict, available_data_dict):
    judgment_result = {}
    index = 1
    length = len(available_data_dict)
    for key,value in available_data_dict.items():
        print(str(index) + '/' + str(length))
        y = []
        i = 0
        data_description = ''
        for key_temp, value_temp in value.items():
            if key_temp in triu_matrix_dict.keys():
                data_description = data_description + key_temp + '_'
                triu_matrix = triu_matrix_dict[key_temp]
                if i == 0:
                    X = triu_matrix
                else:
                    X = np.vstack((X, triu_matrix))
                i = i + 1
                y.append(value_temp)
        y = np.array(y)
        acc_avg = verify_process(X,y)
        print(key, acc_avg)
        judgment_result[key] = float(acc_avg)
        index = index + 1

    with open('judgment_result.json', 'w') as fp:
        json.dump(judgment_result, fp)


def verify_process(X,y):
    s_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    score = 0
    for train_index, test_index in s_folder.split(X,y):
        clf = LogisticRegression(random_state=0).fit(X[train_index], y[train_index])
        score = score + clf.score(X[test_index], y[test_index])
    return score/5


def generate_fc_matrix_without_augmentation(filepath_list, output_path):
    mask_file_path = 'F:\\ComparativeLearning\\BN_Atlas_246_3mm.nii'
    mask = load_nifti(mask_file_path)
    for each_filepath in filepath_list:
        for specific_filepath in os.listdir(each_filepath):
            if len(specific_filepath.split('_')) == 4:
                file_id = specific_filepath.split('_')[0] + '_' + specific_filepath.split('_')[1]
            else:
                file_id = specific_filepath.split('_')[0] + '_' + specific_filepath.split('_')[1] + '_' + specific_filepath.split('_')[2]
            print(file_id)
            specific_filepath = os.path.join(each_filepath, specific_filepath)
            original_data = load_nifti(specific_filepath)
            bold_matrix = voxel_to_bold(file_name=file_id, original_data=original_data, mask=mask, roi_ratio=1)
            fc_matrix = bold_to_fc(file_name=file_id, bold_array=bold_matrix)
            np.savetxt(os.path.join(output_path, file_id+'.txt'),  fc_matrix)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    # filepath_list = [collected_data_dir + '\\' + s for s in os.listdir(collected_data_dir)]
    output_path = 'F:\\AbideData_FC_Matrix'
    # generate_fc_matrix_without_augmentation(filepath_list, output_path)
    selected_facility_list = select_facility()
    df = pd.read_csv(data_description_filepath)
    triu_matrix_dict = triu_matrix_generation(df, output_path)
    print('TRIU MATRIX FINISHED')

    # available_data_dict = get_available_data(selected_facility_list, df, 100)
    # with open('available_data_dict.json', 'w') as fp:
    #     json.dump(available_data_dict, fp)

    f = open('available_data_dict.json', 'r')
    content = f.read()
    available_data_dict = json.loads(content)
    print('Available Data Dict Load Already')

    judge_fc_matrix(triu_matrix_dict, available_data_dict)

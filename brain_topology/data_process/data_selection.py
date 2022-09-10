import json
import os, random
from configparser import ConfigParser

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from data_preprocess import load_txt_into_array

config = ConfigParser()
config.read('../parameters.ini', encoding='UTF-8')


# 该函数用来判断那些数据在逻辑回归中有较好的表现
def data_choose_optimization(try_number):
    cv_number = config.getint('experiment', 'cv_number')
    best_id = []
    best_skf = []
    best_acc = 0

    record = []
    fc_dir = config.get('filepath', 'no_aug_no_aug_dir')
    model = LogisticRegression(C=1, solver='liblinear')
    label_set = []
    id_set = []

    i = 0
    for file in os.listdir(fc_dir):
        print(i)
        file_path = fc_dir + '/' + file
        fc_info = load_txt_into_array(file_path)
        fc_array = fc_info['data']
        fc_label = fc_info['label']
        fc_shape = fc_info['shape']
        fc_id = fc_info['id']
        iu = np.triu_indices(fc_shape[0])
        fc_array = fc_array[iu]
        if i == 0:
            data_set = fc_array
        else:
            data_set = np.vstack((data_set, fc_array))
        id_set.append(fc_id)
        label_set.append(fc_label)
        i = i + 1

    id_set = np.array(id_set)
    label_set = np.array(label_set)
    label_zero = np.where(label_set == 0)[0]
    label_one = np.where(label_set == 1)[0]

    for j in range(try_number):
        choose_index = random.sample(label_zero.tolist(), 50) + random.sample(label_one.tolist(), 50)
        chosen_data = data_set[choose_index, :]
        chosen_label = label_set[choose_index]
        chosen_id = id_set[choose_index]

        skf = StratifiedKFold(n_splits=cv_number).split(chosen_data, chosen_label)
        skf_info = []
        acc = 0
        for train_index, test_index in skf:
            X_train, X_test = chosen_data[train_index], chosen_data[test_index]
            y_train, y_test = chosen_label[train_index], chosen_label[test_index]
            skf_info.append((train_index, test_index))
            model.fit(X_train, y_train)
            acc = acc + model.score(X_test, y_test)
        acc_avg = acc / cv_number
        print(acc_avg)
        record.append({
            'id': chosen_id,
            'acc': acc_avg,
        })

        if acc_avg > best_acc:
            best_acc = acc_avg
            best_id = chosen_id
            best_skf = skf_info
    print('----')
    print(best_acc)
    print(best_id)

    # 数据存储
    cv_info = []
    for i in range(cv_number):
        i_th_train_and_test = best_skf[i]
        i_th_train = i_th_train_and_test[0]
        i_th_test = i_th_train_and_test[1]
        i_th_train_file_id = best_id[i_th_train]
        i_th_test_file_id = best_id[i_th_test]
        cv_info.append({
            'train_file_id': i_th_train_file_id.tolist(),
            'test_file_id': i_th_test_file_id.tolist(),
        })
    with open('../data/encoded_data/no_aug_no_aug/cv_info.json', 'w') as file_object:
        json.dump(cv_info, file_object)


def ml_reappearance():
    cv_number = config.getint('experiment', 'cv_number')

    fc_dir = config.get('filepath', 'no_aug_no_aug_dir')
    model = LogisticRegression(C=1, solver='liblinear')
    label_set = []
    id_set = []

    i = 0
    for file in os.listdir(fc_dir):
        print(i)
        if file.split('.')[1] == 'txt':
            file_path = fc_dir + '/' + file
            fc_info = load_txt_into_array(file_path)
            fc_array = fc_info['data']
            fc_label = fc_info['label']
            fc_shape = fc_info['shape']
            fc_id = fc_info['id']
            iu = np.triu_indices(fc_shape[0])
            fc_array = fc_array[iu]
            if i == 0:
                data_set = fc_array
            else:
                data_set = np.vstack((data_set, fc_array))
            id_set.append(fc_id)
            label_set.append(fc_label)
            i = i + 1
    id_set = np.array(id_set)
    label_set = np.array(label_set)
    with open("../data/cv_info/cv_info.json", 'r') as load_f:
        cv_info = json.load(load_f)
    for i in range(cv_number):
        cv_info_train_file_id = cv_info[i]['train_file_id']
        cv_info_test_file_id = cv_info[i]['test_file_id']
        train_index = []
        test_index = []
        for m in range(len(cv_info_train_file_id)):
            train_index.append(np.where(id_set == cv_info_train_file_id[m])[0][0])
        for n in range(len(cv_info_test_file_id)):
            test_index.append(np.where(id_set == cv_info_test_file_id[n])[0][0])
        train_data_set = data_set[train_index]
        train_label_set = label_set[train_index]

        test_data_set = data_set[test_index]
        test_label_set = label_set[test_index]

        model.fit(train_data_set, train_label_set)
        acc = model.score(test_data_set, test_label_set)
        print(acc)


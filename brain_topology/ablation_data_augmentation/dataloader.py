import os, joblib


def load_data_loaders_from_pkl(file_path):
    data_loaders = joblib.load(open(file_path, 'rb'))
    return data_loaders


def generate_dataloader(dataset_config, each_exp):
    dataloader_dir = dataset_config['dataloader_dir']
    unaug_loader_type = dataset_config['loader_type']

    unaug_train_list = []
    unaug_train_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(each_exp), 'train')
    for loader in os.listdir(unaug_train_loader_path_this_cv):
        unaug_train_list.append(load_data_loaders_from_pkl(os.path.join(unaug_train_loader_path_this_cv, loader)))

    unaug_test_list = []
    unaug_test_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(each_exp), 'test')
    for loader in os.listdir(unaug_test_loader_path_this_cv):
        unaug_test_list.append(load_data_loaders_from_pkl(os.path.join(unaug_test_loader_path_this_cv, loader)))

    return unaug_train_list, unaug_test_list
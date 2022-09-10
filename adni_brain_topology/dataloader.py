import os
import pickle
from torch.utils.data import Dataset


# label2label_dict = {
#
# }

def generate_subject_label_dict(file_path):
    subject_label_dict = pickle.load(open(file_path, 'rb'))
    return subject_label_dict


def load_voxel_from_dir(dir_path):
    file_name_full_path_list = []
    file_name_list = os.listdir(dir_path)
    for each_file_name in file_name_list:
        subject_id = each_file_name.split('.')[0]
        each_file_name_full_path = os.path.join(dir_path, each_file_name)
        file_name_full_path_list.append((subject_id, each_file_name_full_path))
    return file_name_full_path_list


class ADNIDataset(Dataset):
    def __init__(self, graph_label_pickle):
        self.dataset_list = pickle.load(open(graph_label_pickle,'rb'))['graph']
        self.label_list = pickle.load(open(graph_label_pickle,'rb'))['label']


    def __getitem__(self, index):
        return self.dataset_list[index], self.label_list[index]

    def __len__(self):
        return len(self.dataset_list)




import pickle

label_2_label_dict = {

}


def load_subject_label():
    subject_labels_dict_filepath = "F:\\ADNI\\ADNI_data\\subject_labels_dict.pkl"
    file = open(subject_labels_dict_filepath, 'rb')
    subject_labels_dict = pickle.load(file)
    return subject_labels_dict


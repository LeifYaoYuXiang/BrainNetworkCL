import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import os
from sklearn.manifold import TSNE
from data_process.data_augment import load_nifti, voxel_to_bold
from data_process.data_preprocess import load_data_loaders_from_pkl
from utils import load_model


def display_matrix_2d(mat):
    plt.matshow(mat, vmin=0, vmax=99)
    plt.show()


def display_matrix_3d(mat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    w = [i+1 for i in range(mat.shape[0])]
    q = [i+1 for i in range(mat.shape[1])]
    W,Q = np.meshgrid(w,q)

    ax.set_xlabel('x_metric')
    ax.set_ylabel('y_metric')
    ax.set_zlabel('value')
    surf = ax.plot_surface(W, Q, mat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def display_time_series(mask_file_path, voxel_filepath):
    mask = load_nifti(mask_file_path)
    original_data = load_nifti(voxel_filepath)
    bold_matrix = voxel_to_bold(file_name='draw', original_data=original_data, mask=mask, roi_ratio=1)
    average_bold_time_series = np.mean(bold_matrix, axis=0)
    average_bold_time_series = average_bold_time_series[0].tolist()[0]
    x = range(len(average_bold_time_series))
    plt.plot(x, average_bold_time_series, 'r--')

    plt.title('BOLD Time Series')
    plt.show()


def t_sne_visualization(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        if label[i] == 0:
            plt.plot(data[i, 0], data[i, 1], 'o', color='b')
        else:
            plt.plot(data[i, 0], data[i, 1], 'o', color='g')
    plt.xticks()
    plt.yticks()
    plt.show()
    return fig


def generate_t_sne_embedding(data, label):
    ts = TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(data)
    fig = t_sne_visualization(result, label)
    plt.show()


def get_embedding(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i in range(len(dataloader)):
            graph_info = dataloader[i]
            graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
            batch_size = graph_info['batch_size']
            graph = graph_info['batch_graph']
            input = (graph_node_features, graph, batch_size)
            embedding = model.get_embedding(input)
            if i == 0:
                data = embedding.numpy()
                label = np.array(graph_info['batch_label'])
            else:
                data = np.vstack((data, embedding.numpy()))
                label = np.append(label, np.array(graph_info['batch_label']))
    return data, label



if __name__ == '__main__':
    filepath = 'run\\2022-03-17T20-57-13\\0cv_.model.pkl'
    each_exp = 0
    n = 100
    dataloader_dir = 'F:\\OHComparativeLearning\\dataloader'
    train1_loader_type = 'aug_slide_window'
    train2_loader_type = 'no_aug_ratio_sample'
    unaug_loader_type = 'no_aug_no_aug'

    train_loader1_path_list = []
    train_loader1_path_this_cv = os.path.join(dataloader_dir, train1_loader_type, str(each_exp), 'train')
    for loader in os.listdir(train_loader1_path_this_cv):
        train_loader1_path_list.append(os.path.join(train_loader1_path_this_cv, loader))

    train_loader2_path_list = []
    train_loader2_path_this_cv = os.path.join(dataloader_dir, train2_loader_type, str(each_exp), 'train')
    for loader in os.listdir(train_loader2_path_this_cv):
        train_loader2_path_list.append(os.path.join(train_loader2_path_this_cv, loader))

    unaug_train_loader_path_list = []
    unaug_train_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(each_exp), 'test')
    for loader in os.listdir(unaug_train_loader_path_this_cv):
        unaug_train_loader_path_list.append(os.path.join(unaug_train_loader_path_this_cv, loader))

    train_loader_path_this_epoch = train_loader1_path_list[n]
    train_loader2_path_this_epoch = train_loader2_path_list[n]
    unaug_train_loader_path_this_epoch = unaug_train_loader_path_list[n]
    train_loader_this_epoch = load_data_loaders_from_pkl(train_loader_path_this_epoch)
    train_loader2_this_epoch = load_data_loaders_from_pkl(train_loader2_path_this_epoch)
    unaug_train_loader_this_epoch = load_data_loaders_from_pkl(unaug_train_loader_path_this_epoch)
    dataloader = train_loader_this_epoch + train_loader2_this_epoch
    # dataloader = unaug_train_loader_this_epoch
    model = load_model(filepath)[0]
    data, label = get_embedding(model, dataloader)
    generate_t_sne_embedding(data, label)



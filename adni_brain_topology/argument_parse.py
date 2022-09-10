import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="heroGraph arguments")

    # dataset configuration
    parser.add_argument('--train1_loader_type', type=str, default='NoAug_RatioSample')
    parser.add_argument('--train2_loader_type', type=str, default='Aug_SlideWindow')
    parser.add_argument('--unaug_loader_type', type=str, default='NoAug_NoAug')
    parser.add_argument('--dataloader_dir', type=str, default='/home/xuesong/yuxiang_yao/brain_topology_data/ADNI/ADNI_data/dataloader_binary')

    # model configuration
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_mlp_layers', type=int, default=3)
    parser.add_argument('--in_feats', type=int, default=246)
    parser.add_argument('--n_hidden', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--node_each_graph', type=int, default=246)
    parser.add_argument('--learn_eps', type=bool, default=True)
    parser.add_argument('--graph_pooling_type', type=str, default='sum')
    parser.add_argument('--neighbor_pooling_type', type=str, default='sum')
    parser.add_argument('--final_dropout', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.2)

    # training configuration
    parser.add_argument('--cv_number', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--ss_n_epoch', type=int, default=200)
    parser.add_argument('--ss_lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nt_xent_loss_temperature', type=float, default=1)
    parser.add_argument('--log_filepath', type=str, default='run')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=1024, help='random seed for training and testing')
    parser.add_argument('--comment', type=str, default='default comment', help='comment for each experiment')
    args = parser.parse_args()
    return args



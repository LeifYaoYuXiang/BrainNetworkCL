import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="OH ablation arguments")

    # dataset configuration
    parser.add_argument('--data_pkl_filepath', type=str, default='data/dataloader/dataloader.pkl')
    parser.add_argument('--train_test_ratio', type=float, default=0.7)

    # model configuration
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--in_feats', type=int, default=246)
    parser.add_argument('--n_hidden', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--node_each_graph', type=int, default=246)
    parser.add_argument('--dropout', type=float, default=0.2)

    # training configuration
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--nt_xent_loss_temperature', type=float, default=1)
    parser.add_argument('--log_filepath', type=str, default='run')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=1024, help='random seed for training and testing')
    parser.add_argument('--comment', type=str, default='default comment', help='comment for each experiment')
    args = parser.parse_args()
    return args
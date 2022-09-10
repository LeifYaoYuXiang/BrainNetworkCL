from configparser import ConfigParser

import torch

from model_v_gin import Model
from pytorch_metric_learning.losses import NTXentLoss
from sklearn.linear_model import LogisticRegression

from argument_parse import parse_args
from dataloader import generate_dataloader
from train_and_test import train_and_eval_ablation
from utils import seed_setting, get_summary_writer, record_configuration


def main(args):
    config = ConfigParser()
    config.read('../parameters.ini', encoding='UTF-8')
    dataloader_dir = config.get('filepath', 'dataloader_dir')

    dataset_config = {
        'dataloader_dir': dataloader_dir,
        'loader_type': args.loader_type,
    }

    model_config = {
        'n_layers': args.n_layers,
        'n_mlp_layers': args.n_mlp_layers,
        'in_feats': args.in_feats,
        'n_hidden': args.n_hidden,
        'node_each_graph': args.node_each_graph,
        'n_classes': args.n_classes,
        'final_dropout': args.final_dropout,
        'learn_eps': args.learn_eps,
        'graph_pooling_type': args.graph_pooling_type,
        'neighbor_pooling_type': args.neighbor_pooling_type
    }

    train_test_config = {
        'cv_number': args.cv_number,
        'n_epoch': args.n_epoch,
        'nt_xent_loss_temperature': args.nt_xent_loss_temperature,
        'lr': args.lr,
        'device': args.device,
    }

    seed = args.seed
    log_filepath = args.log_filepath

    seed_setting(seed)

    summary_writer, log_dir = get_summary_writer(log_filepath)

    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_test_config,
    })
    # 加载数据集
    print('开始加载我们的数据')
    dataloader_cv_dict = {}
    for each_exp in range(args.cv_number):
        dataloader_cv_dict[each_exp] = generate_dataloader(dataset_config, each_exp)
        print('第' + str(each_exp+1) + '折数据加载完成')

    # 加载模型 & 相关的优化器
    model = Model(model_config)
    ml_model = LogisticRegression()
    loss_fcn = NTXentLoss(temperature=train_test_config['nt_xent_loss_temperature'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])
    train_and_eval_ablation(train_test_config, dataloader_cv_dict, model, ml_model, loss_fcn, optimizer, summary_writer, log_dir)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)
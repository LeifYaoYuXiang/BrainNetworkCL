import random

import torch
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn
from torch.optim.lr_scheduler import StepLR

from argument_parser import parse_args
from dataloader import build_dataloader
from model_v1 import Model
from train_eval_test_v1 import train_test_eval_v1
from utils import seed_setting, get_summary_writer, record_configuration


def main(args):
    dataset_config = {
        'data_dir': args.data_dir,
        'data_length': args.data_length,
        'total_epoch': args.total_epoch,
        'train_test_ratio': args.train_test_ratio,
        'unaug_method': args.unaug_method,
        'aug_1_method': args.aug_1_method,
        'aug_2_method': args.aug_2_method,
        'device': args.device,
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
        'n_epoch': args.n_epoch,
        'cl_n_epoch': args.cl_n_epoch,
        'lr': args.lr,
        'cl_lr': args.cl_lr,
        'nt_xent_loss_temperature': args.nt_xent_loss_temperature,
        'batch_size': args.batch_size,
        'device': args.device,
        'comment': args.comment,
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
    # build dataset
    data_length = dataset_config['data_length']
    train_test_ratio = dataset_config['train_test_ratio']
    train_id_list = random.sample(list(range(data_length)), int(data_length*train_test_ratio))
    eval_id_list = [each_id for each_id in list(range(data_length)) if each_id not in train_id_list]
    unaug_dataset_config = {
        'data_dir': dataset_config['data_dir'],
        'device': dataset_config['device'],
        'total_epoch': dataset_config['total_epoch'],
        'aug_method': dataset_config['unaug_method'],
        'train_id_list': train_id_list,
        'eval_id_list': eval_id_list,
    }
    unaug_train_dataloader, unaug_eval_dataloader = build_dataloader(unaug_dataset_config)

    aug_1_dataset_config = {
        'data_dir': dataset_config['data_dir'],
        'device': dataset_config['device'],
        'total_epoch': dataset_config['total_epoch'],
        'aug_method': dataset_config['aug_1_method'],
        'train_id_list': train_id_list,
        'eval_id_list': eval_id_list,
    }
    aug_1_train_dataloader, aug_1_eval_dataloader = build_dataloader(aug_1_dataset_config)

    aug_2_dataset_config = {
        'data_dir': dataset_config['data_dir'],
        'device': dataset_config['device'],
        'total_epoch': dataset_config['total_epoch'],
        'aug_method': dataset_config['aug_2_method'],
        'train_id_list': train_id_list,
        'eval_id_list': eval_id_list,
    }
    aug_2_train_dataloader, aug_2_eval_dataloader = build_dataloader(aug_2_dataset_config)
    # build model
    model = Model(model_config)
    if train_test_config['device'] == 'gpu':
        model.to(torch.device('cuda:0'))

    cl_loss_fcn = NTXentLoss(temperature=train_test_config['nt_xent_loss_temperature'])
    loss_fcn = nn.CrossEntropyLoss()

    cl_optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['cl_lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])

    cl_scheduler = StepLR(cl_optimizer, step_size=20, gamma=0.1)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # start training & evaluation
    train_test_eval_v1(unaug_train_dataloader, unaug_eval_dataloader, aug_1_train_dataloader, aug_2_train_dataloader,
                       model, cl_optimizer, optimizer, cl_loss_fcn, loss_fcn, cl_scheduler, scheduler,
                       train_test_config, summary_writer)


# tensorboard --logdir=run --port 4445
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)
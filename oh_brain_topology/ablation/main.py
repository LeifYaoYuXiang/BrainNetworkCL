import torch
from torch import nn
from pytorch_metric_learning.losses import NTXentLoss
from torch.optim.lr_scheduler import StepLR

from ablation.argument_parser import parse_args
from ablation.dataloader_v1 import build_dataloader
from ablation.model_v1 import GraphEnocder_v1
from ablation.train_eval_test_v1 import train_test_eval_v1
from ablation.utils import record_configuration, seed_setting, get_summary_writer


def main(args):
    dataset_config = {
        'pkl_filepath': args.data_pkl_filepath,
        'train_test_ratio': args.train_test_ratio,
        'device': args.device,
    }
    model_config = {
        'n_layers': args.n_layers,
        'in_feats': args.in_feats,
        'n_hidden': args.n_hidden,
        'node_each_graph': args.node_each_graph,
        'n_classes': args.n_classes,
        'dropout': args.dropout,
    }
    train_test_config = {
        'aug_1_method': args.aug_1_method,
        'aug_2_method': args.aug_2_method,
        'n_epoch': args.n_epoch,
        'cl_n_epoch': args.cl_n_epoch,
        'lr': args.lr,
        'cl_lr': args.cl_lr,
        'nt_xent_loss_temperature': args.nt_xent_loss_temperature,
        'batch_size': args.batch_size,
        'device': args.device,
        'comment': args.comment,
    }
    print(train_test_config['aug_1_method'], train_test_config['aug_2_method'])
    seed = args.seed
    log_filepath = args.log_filepath
    seed_setting(seed)
    summary_writer, log_dir = get_summary_writer(log_filepath)
    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_test_config,
    })

    train_dataloader, eval_dataloader = build_dataloader(dataset_config)

    model = GraphEnocder_v1(model_config)
    if train_test_config['device'] == 'gpu':
        model.to(torch.device('cuda:0'))
    cl_loss_fcn = NTXentLoss(temperature=train_test_config['nt_xent_loss_temperature'])
    loss_fcn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    cl_optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['cl_lr'])
    cl_scheduler = StepLR(cl_optimizer, step_size=20, gamma=0.1)

    train_test_eval_v1(train_dataloader, eval_dataloader,
                       model, cl_optimizer, optimizer, cl_loss_fcn, loss_fcn, cl_scheduler, scheduler,
                       train_test_config, summary_writer)

# cd ablation
# tensorboard --logdir=run --port 4444
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)
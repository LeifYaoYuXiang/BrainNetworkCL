import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from benchmark.dataloader_v1 import build_dataloader
from benchmark.argument_parser import parse_args
from benchmark.train_eval_test_v1 import train_test_eval_v1
from benchmark.utils import seed_setting, get_summary_writer, record_configuration
from benchmark.model_v1 import MedianModel
# from benchmark.model_v2 import AdaptiveGNN


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
        'model': 'MedianModel',
        'n_epoch': args.n_epoch,
        'lr': args.lr,
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

    train_dataloader, eval_dataloader = build_dataloader(dataset_config)

    model = MedianModel(model_config)
    # model = AdaptiveGNN(model_config)
    if train_test_config['device'] == 'gpu':
        model.to(torch.device('cuda:0'))
    loss_fcn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_test_config['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_test_eval_v1(train_dataloader, eval_dataloader, model, optimizer, loss_fcn, scheduler, train_test_config, summary_writer)
    print('finish')


# tensorboard --logdir=run --port 4444
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)

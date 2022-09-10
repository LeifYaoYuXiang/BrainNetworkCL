# from train_and_test import train_and_eval
from train_and_test_v2 import train_and_eval_v2
from argument_parse import parse_args
from utils import seed_setting, get_summary_writer, record_configuration


def main(args):
    dataset_config = {
        'dataloader_dir': args.dataloader_dir,
        'train1_loader_type': args.train1_loader_type,
        'train2_loader_type': args.train2_loader_type,
        'unaug_loader_type': args.unaug_loader_type,
    }
    model_config = {
        'n_layers': args.n_layers,
        'n_mlp_layers': args.n_mlp_layers,
        'in_feats': args.in_feats,
        'n_hidden': args.n_hidden,
        'node_each_graph': args.node_each_graph,
        'n_classes': args.n_classes,
        'final_dropout': args.final_dropout,
        'dropout': args.dropout,
        'learn_eps': args.learn_eps,
        'graph_pooling_type': args.graph_pooling_type,
        'neighbor_pooling_type': args.neighbor_pooling_type
    }
    train_test_config = {
        'cv': args.cv_number,
        'n_epoch': args.n_epoch,
        'ss_n_epoch': args.ss_n_epoch,
        'nt_xent_loss_temperature': args.nt_xent_loss_temperature,
        'lr': args.lr,
        'ss_lr': args.ss_lr,
        'device': args.device,
        'comment': 'without freezing parameters'
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
    # train_and_eval(train_test_config, dataset_config, model_config, summary_writer, log_dir)
    train_and_eval_v2(train_test_config, dataset_config, model_config, summary_writer, log_dir)


# tensorboard --logdir=run --port 4444
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)


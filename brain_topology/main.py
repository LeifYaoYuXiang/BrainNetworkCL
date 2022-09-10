from configparser import ConfigParser
from argument_parse import parse_args
from utils import seed_setting, get_summary_writer, record_configuration
from train_test_v1 import train_and_eval


def main(args):
    config = ConfigParser()
    config.read('parameters.ini', encoding='UTF-8')
    dataloader_dir = config.get('filepath', 'dataloader_dir')

    dataset_config = {
        'dataloader_dir': dataloader_dir,
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

    train_and_eval(train_test_config, dataset_config, model_config, summary_writer, log_dir)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)

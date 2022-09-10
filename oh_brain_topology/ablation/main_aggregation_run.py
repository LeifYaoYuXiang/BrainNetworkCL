from ablation.argument_parser import parse_args
from ablation.main import main
from ablation.utils import send_notification_email


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    aug_methods = ['dropping_node', 'dropping_edge', 'subgraph_generation', 'feat_masking']
    args = parse_args()
    for each_aug_method in aug_methods:
        args.aug_1_method = each_aug_method
        args.aug_2_method = each_aug_method
        print(each_aug_method, each_aug_method)
        main(args)
        print('finish')


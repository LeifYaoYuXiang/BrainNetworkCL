from argument_parser import parse_args
from main import main
no_aug_description = 'phase_1_no_aug_phase_2_no_aug'
available_phase_1_augmentation_list = ['phase_1_no_aug', 'phase_1_voxel_sampling', 'phase_1_voxel_radial_sampling']
available_phase_2_augmentation_list = ['phase_2_no_aug', 'phase_2_slide_window', 'phase_2_ratio_sample']


def build_augmentation_list():
    augmentation_list = []
    for each_available_phase_1_augmentation in available_phase_1_augmentation_list:
        for each_available_phase_2_augmentation in available_phase_2_augmentation_list:
            augmentation_list.append(each_available_phase_1_augmentation + '_' + each_available_phase_2_augmentation)
    return augmentation_list


# def main_aggregation(augmentation_list):
#     for aug_1 in augmentation_list:
#         for aug_2 in augmentation_list:
#             if aug_1 != no_aug_description and aug_2 != no_aug_description and aug_1 != aug_2:
#                 print(aug_1, aug_2)
#                 args = parse_args()
#                 args.aug_1_method = aug_1
#                 args.aug_2_method = aug_2
#                 print(args)
#                 main(args)
#                 print('finsih')


def main_aggregation(augmentation_list):
    for aug in augmentation_list:
        print(aug, aug)
        args = parse_args()
        args.aug_1_method = aug
        args.aug_2_method = aug
        print(args)
        main(args)
        print('finsih')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    augmentation_list = build_augmentation_list()
    main_aggregation(augmentation_list)
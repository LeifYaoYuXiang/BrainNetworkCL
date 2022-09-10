import os
import configparser
from tensorboard.backend.event_processing import event_accumulator
ini_file = 'configuration.ini'


def findMaxAverage(nums, k: int):
    low = 0
    all = sum(nums[0:k])
    avg = all
    for i in range(k, len(nums)):
        all = all + nums[i] - nums[low]
        low += 1
        avg = max(avg, all)
    return avg / k


def tensorboard_reading(run_dir):
    k = 10
    exp_dir = os.listdir(run_dir)
    acc_exp_result = {}
    f1_exp_result = {}
    for each_exp_dir in exp_dir:
        # 读取 Configuration
        parser = configparser.ConfigParser()
        parser.read(os.path.join(run_dir, each_exp_dir, ini_file))
        aug_1_method = parser.get('DATASET', 'aug_1_method')
        aug_2_method = parser.get('DATASET', 'aug_2_method')
        # 读取 Tensorboard 数据
        ea = event_accumulator.EventAccumulator(os.path.join(run_dir, each_exp_dir, os.listdir(os.path.join(run_dir, each_exp_dir))[-1]))
        ea.Reload()
        acc = ea.scalars.Items('Acc')
        f1 = ea.scalars.Items('F1')
        if (aug_1_method, aug_2_method) not in acc_exp_result.keys() and (aug_2_method, aug_1_method) not in acc_exp_result.keys():
            acc_exp_result[(aug_1_method, aug_2_method)] = findMaxAverage([i.value for i in acc], k)
            f1_exp_result[(aug_1_method, aug_2_method)] = findMaxAverage([i.value for i in f1], k)
    print(sorted(acc_exp_result.items(), key=lambda kv:(kv[1], kv[0])))
    print(sorted(f1_exp_result.items(), key=lambda kv:(kv[1], kv[0])))


if __name__ == '__main__':
    run_dir = r'D:\PycharmProjects\oh_brain_topology\run'
    tensorboard_reading(run_dir)

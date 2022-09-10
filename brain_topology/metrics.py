import torch


def acc_metric(indices_record, batch_y_record):
    correct = torch.sum(indices_record == batch_y_record)
    acc_score = correct.item() * 1.0 / len(batch_y_record)
    return acc_score


def precision_metric(indices_record, batch_y_record):
    a = indices_record == batch_y_record
    b = indices_record == 1
    tp = torch.sum(a & b)
    tp_fp = torch.sum(indices_record == 1)

    if tp_fp.item() * 1.0 != 0:
        return tp.item() * 1.0 / tp_fp.item() * 1.0
    else:
        return 0


def recall_metric(indices_record, batch_y_record):
    a = indices_record == batch_y_record
    b = batch_y_record == 1
    tp = torch.sum(a & b)
    tp_fn = torch.sum(batch_y_record == 1)
    if tp_fn.item() * 1.0 != 0:
        return tp.item() * 1.0 / tp_fn.item() * 1.0
    else:
        return 0


# 计算f1的metric
def f1_metric(precision, recall):
    if precision + recall != 0:
        return (2*precision*recall) / (precision + recall)
    else:
        return 0
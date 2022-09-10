import torch
from sklearn.metrics import accuracy_score, f1_score


def train_test_eval_v1(train_dataloader, eval_dataloader,
                       model, optimizer, loss_fcn, scheduler,
                       train_test_config, summary_writer):
    n_epoch = train_test_config['n_epoch']
    batch_size = train_test_config['batch_size']
    device = train_test_config['device']
    for each_epoch in range(n_epoch):
        # train the model
        labeled_loss_value = 0.0
        train_epoch_data = train_dataloader.get_batch(batch_size)
        # [([batch_graph, batch_graph], [batch_label, batch_label]),  ... , ...]
        idx = 0
        for each_batch_data in train_epoch_data:
            train_batch_graph = each_batch_data[0]
            train_batch_label = each_batch_data[1]
            train_index = 0
            for each_graph in train_batch_graph:
                if train_index == 0:
                    train_batch_embedding = model(each_graph)
                else:
                    train_batch_embedding = torch.concat((train_batch_embedding, model(each_graph)),  0)
                train_index = train_index + 1
            if device == 'cpu':
                train_batch_label = torch.LongTensor(train_batch_label)
            else:
                train_batch_label = torch.LongTensor(train_batch_label).cuda()
            label_loss = loss_fcn(train_batch_embedding, train_batch_label)
            optimizer.zero_grad()
            label_loss.backward()
            optimizer.step()
            labeled_loss_value = labeled_loss_value + label_loss.item()
            print(idx)
            idx = idx + 1
        # eval the model
        with torch.no_grad():
            eval_epoch_logits = []
            eval_epoch_label = []
            eval_epoch_data = eval_dataloader.get_batch(batch_size)
            # [([batch_graph], [batch_label]),  ... , ...]
            for each_batch_data in eval_epoch_data:
                eval_batch_graph = each_batch_data[0]
                eval_batch_label = each_batch_data[1]
                eval_index = 0
                for each_graph in eval_batch_graph:
                    if eval_index == 0:
                        eval_batch_logits = model(each_graph)
                    else:
                        eval_batch_logits = torch.concat((eval_batch_logits, model(each_graph)), 0)
                    eval_index = eval_index + 1
                eval_batch_logits = eval_batch_logits.argmax(1).tolist()
                eval_epoch_logits = eval_epoch_logits + eval_batch_logits
                eval_epoch_label = eval_epoch_label + eval_batch_label
        acc = accuracy_score(eval_epoch_logits, eval_epoch_label)
        f1 = f1_score(eval_epoch_logits, eval_epoch_label)
        summary_writer.add_scalar('labeled_loss', labeled_loss_value, each_epoch)
        summary_writer.add_scalar('Acc', acc, each_epoch)
        summary_writer.add_scalar('F1', f1, each_epoch)
        print('labeled learning', each_epoch, labeled_loss_value, acc, f1)
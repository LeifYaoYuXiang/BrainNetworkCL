import torch
from sklearn.metrics import accuracy_score, f1_score

from data_preprocess import build_dgl_graph


def train_test_eval_v1(unaug_train_dataloader, unaug_eval_dataloader, aug_1_train_dataloader, aug_2_train_dataloader,
                       model, cl_optimizer, optimizer, cl_loss_fcn, loss_fcn, cl_scheduler, scheduler,
                       train_test_config, summary_writer):
    n_epoch = train_test_config['n_epoch']
    cl_n_epoch = train_test_config['cl_n_epoch']
    batch_size = train_test_config['batch_size']
    device = train_test_config['device']

    # contrastive learning for pretraining
    for each_epoch in range(cl_n_epoch):
        cl_loss_val = 0.0
        aug_1_batch_dataloader_each_epoch = aug_1_train_dataloader.get_batch(each_epoch, batch_size)
        aug_2_batch_dataloader_each_epoch = aug_2_train_dataloader.get_batch(each_epoch, batch_size)
        assert len(aug_1_batch_dataloader_each_epoch) == len(aug_2_batch_dataloader_each_epoch)
        for i in range(len(aug_1_batch_dataloader_each_epoch)):
            aug_1_batch_graph_data = aug_1_batch_dataloader_each_epoch[i][0]
            aug_2_batch_graph_data = aug_2_batch_dataloader_each_epoch[i][0]
            assert len(aug_1_batch_graph_data) == len(aug_2_batch_graph_data)
            batch_index = 0
            for j in range(len(aug_1_batch_graph_data)):
                if batch_index == 0:
                    # for bug version
                    aug_1_embedding = model.get_embedding(build_dgl_graph(aug_1_batch_graph_data[j]))
                    aug_2_embedding = model.get_embedding(build_dgl_graph(aug_2_batch_graph_data[j]))
                    # for non-bug version
                    # aug_1_embedding = model.get_embedding(aug_1_batch_graph_data[j])
                    # aug_2_embedding = model.get_embedding(aug_2_batch_graph_data[j])
                else:
                    # for bug version
                    aug_1_embedding = torch.concat((aug_1_embedding, model.get_embedding(build_dgl_graph(aug_1_batch_graph_data[j]))), 0)
                    aug_2_embedding = torch.concat((aug_2_embedding, model.get_embedding(build_dgl_graph(aug_2_batch_graph_data[j]))), 0)
                    # for non-bug version
                    # aug_1_embedding = torch.concat((aug_1_embedding, model.get_embedding(aug_1_batch_graph_data[j])), 0)
                    # aug_2_embedding = torch.concat((aug_2_embedding, model.get_embedding(aug_2_batch_graph_data[j])), 0)
                batch_index = batch_index + 1
            aug_embeddings = torch.cat((aug_1_embedding, aug_2_embedding))
            indices = torch.arange(batch_index)
            if device == 'cpu':
                label = torch.cat((indices, indices))
            else:
                label = torch.cat((indices, indices)).cuda()
            cl_loss = cl_loss_fcn(aug_embeddings, label)
            cl_loss_val = cl_loss_val + cl_loss.item()
            cl_optimizer.zero_grad()
            cl_loss.backward()
            cl_optimizer.step()
        cl_scheduler.step()
        summary_writer.add_scalar('cl_loss', cl_loss_val, each_epoch)
        print('contrastive learning', each_epoch, cl_loss_val)

    # labeled learning for supervised learning
    for each_epoch in range(n_epoch):
        labeled_loss_value = 0.0
        aug_1_train_dataloader_each_epoch = aug_1_train_dataloader.get_batch(each_epoch, batch_size)
        aug_2_train_dataloader_each_epoch = aug_2_train_dataloader.get_batch(each_epoch, batch_size)
        unaug_train_dataloader_each_epoch = unaug_train_dataloader.get_batch(each_epoch, batch_size)
        assert len(aug_1_train_dataloader_each_epoch) == \
               len(aug_2_train_dataloader_each_epoch) == \
               len(unaug_train_dataloader_each_epoch)
        total_batch_dataloader_each_epoch = aug_1_train_dataloader_each_epoch + aug_2_train_dataloader_each_epoch + unaug_train_dataloader_each_epoch
        for each_batch_data in total_batch_dataloader_each_epoch:
            train_batch_graph = each_batch_data[0]
            train_batch_label = each_batch_data[1]
            train_index = 0
            for each_graph in train_batch_graph:
                # # for non-bug version
                # if train_index == 0:
                #     train_batch_embedding = model(each_graph)
                # else:
                #     train_batch_embedding = torch.concat((train_batch_embedding, model(each_graph)), 0)
                # for bug version
                if train_index == 0:
                    train_batch_embedding = model(build_dgl_graph(each_graph))
                else:
                    train_batch_embedding = torch.concat((train_batch_embedding, model(build_dgl_graph(each_graph))), 0)
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

        # eval the model
        with torch.no_grad():
            eval_epoch_logits = []
            eval_epoch_label = []
            unaug_eval_dataloader_each_epoch = unaug_eval_dataloader.get_batch(each_epoch, batch_size)
            for each_batch_data in unaug_eval_dataloader_each_epoch:
                eval_batch_graph = each_batch_data[0]
                eval_batch_label = each_batch_data[1]
                eval_index = 0
                for each_graph in eval_batch_graph:
                    if eval_index == 0:
                        eval_batch_logits = model(build_dgl_graph(each_graph))
                    else:
                        eval_batch_logits = torch.concat((eval_batch_logits, model(build_dgl_graph(each_graph))), 0)
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
        scheduler.step()

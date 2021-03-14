import torch
import numpy as np
from utils import log_metric
from dataset_cifar10 import CIFAR10


def evaluate(data_loader, model, epoch, cfg,  mode='train', with_logs=True, with_print=True):
    if with_print:
        print(f'Evaluate on {mode} data')
    model.eval()
    nb_classes = data_loader.dataset.nb_classes
    conf_matrix = np.zeros((nb_classes, nb_classes))
    accuracy_sum = 0

    total_iter = len(data_loader)

    for iter, batch in enumerate(data_loader):
        images = batch[0].to(cfg.device)
        labels = batch[1].type(torch.LongTensor).to(cfg.device)

        with torch.no_grad():
            output = model(images)
        _, predict = torch.max(output,1)
        y_pred, y_true = predict.cpu().numpy(), labels.cpu().numpy()
        accuracy_sum += np.sum(y_pred == y_true)
        if cfg.plot_confusion_matrix:
            for i in range(len(images)):
                conf_matrix[y_true[i], y_pred[i]] += 1

        if iter % 50 == 0:
            if with_print:
                print(f'Epoch: {epoch}. Batchs {iter} of {total_iter}.')

    accuracy = accuracy_sum / len(data_loader.dataset)
    if with_print:
        print(f'Epoch:{epoch}. {mode} accuracy {accuracy}')
    if with_logs:
        log_metric(f'eval/accuracy_{mode}',epoch,accuracy,cfg)

    if cfg.logger is not None:
        labels = data_loader.dataset.name_classes
        cfg.logger.report_scalar('eval', f'accuracy_{mode}',accuracy,epoch)
        if cfg.plot_confusion_matrix:
            cfg.logger.report_matrix('ConfusionMatrix',f'Epoch: {epoch}',conf_matrix,epoch,xlabels=labels,ylabels=labels)
    return accuracy


def test(model, cfg,epoch=None, with_print=True):
    ds_test = CIFAR10(cfg.path_to_dataset, cfg, work_mode='test', transform_mode='test')
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)
    return evaluate(dl_test,model,epoch,cfg,mode='test',with_logs=False,with_print=with_print)

if __name__ == '__main__':
    print('Evaluate funcs module')
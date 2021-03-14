import torch
import os
from dataset_cifar10 import CIFAR10
from utils import load_model, log_metric
from torch.utils.data import DataLoader
from test import evaluate
from torchsummary import summary



def train(model, criterion, optimizer, cfg, manual_load=None):
    ds_train = CIFAR10(cfg.path_to_dataset, cfg, transform_mode='train')
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)

    if cfg.evaluate_on_validation_data:
        ds_valid = CIFAR10(cfg.path_to_dataset, cfg, work_mode='valid', transform_mode='test')
        dl_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=True)
    else:
        ds_test = CIFAR10(cfg.path_to_dataset, cfg, work_mode='test', transform_mode='test')
        dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    if cfg.load_save:
        model, optimizer, start_epoch, max_accuracy = load_model(os.path.join(cfg.path_to_saves,cfg.name_save),
                                                                 model,
                                                                 cfg,
                                                                 optimizer)

    elif manual_load is not None:
        optimizer = manual_load['optimizer']
        start_epoch = manual_load['epoch']
        max_accuracy = manual_load['max_accuracy']

    else:
        start_epoch = 0
        max_accuracy = 0
    if cfg.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-8,
                                                                  patience=cfg.ROP_patience,
                                                              factor=cfg.ROP_factor)
        optimizer.zero_grad()
        optimizer.step()

    for epoch in range(start_epoch, cfg.total_epochs):
        training_epoch(dl_train, model, criterion,optimizer,epoch,cfg)
        if cfg.evaluate_on_train_data:
            evaluate(dl_train, model, epoch,cfg,mode='train')

        if cfg.evaluate_on_validation_data:
            accuracy = evaluate(dl_valid, model, epoch, cfg, mode='validation')
        else:
            accuracy = evaluate(dl_test, model, epoch, cfg, mode='test')
        if cfg.use_lr_scheduler:
            lr_scheduler.step(accuracy)

            log_metric(f'loss/learning_rate', epoch, optimizer.param_groups[0]['lr'],cfg)
            if cfg.logger is not None:
                cfg.logger.report_scalar('learning_rate', 'all', optimizer.param_groups[0]['lr'], epoch)

        if cfg.save_models or cfg.save_best_model:
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_accuracy': max_accuracy,
                'opt': optimizer.state_dict()
            }
            if cfg.save_models:
                torch.save(state, os.path.join(cfg.path_to_saves,f'checkpoint_{epoch}'))

            if cfg.save_best_model and max_accuracy < accuracy:
                torch.save(state, os.path.join(cfg.path_to_saves,f'best_checkpoint'))
                max_accuracy = accuracy

            if os.path.exists(os.path.join(cfg.path_to_saves,f'checkpoint_{epoch - 3}')):
                os.remove(os.path.join(cfg.path_to_saves,f'checkpoint_{epoch - 3}'))


def training_epoch(data_loader, model, criterion, optimizer,epoch, cfg):
    print('Train')
    model.train()
    total_iter = len(data_loader)
    for iter, batch in enumerate(data_loader):
        images = batch[0].to(cfg.device)
        labels = batch[1].type(torch.LongTensor).to(cfg.device)


        output = model(images)
        loss = criterion(output, labels, reduction='mean')
        log_metric('loss/train_loss_finetune', total_iter*epoch+iter, loss.item(), cfg)
        if cfg.logger is not None:
            cfg.logger.report_scalar('loss','train_loss_finetune',loss.item(),total_iter*epoch+iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 50 == 0:
            print(f'Epoch: {epoch}. Iteration {iter} of {total_iter}.')



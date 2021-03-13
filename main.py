from config import cfg
import torch.nn.functional as F
from model import ResNet20
from pruning import *
import os
import clearml
from train import train
from test import test
from utils import load_model
from pruning import clustering_filters_models


def main():
    if cfg.clearml_logging:
        with open(cfg.path_to_keys, 'r') as f:
            key, secret = f.readlines()
            key = key[:-1]
        clearml.Task.set_credentials(api_host='https://api.community.clear.ml',
                                 web_host='https://app.community.clear.ml',
                                 files_host='https://files.community.clear.ml',
                                 key=key,
                                 secret=secret)
        task = clearml.Task.init(project_name=cfg.project_name, task_name=cfg.task_name, continue_last_task=True)
        task.connect(cfg)
        task.set_initial_iteration(0)

    cfg.logger = task.get_logger() if cfg.clearml_logging else None
    model = ResNet20().to(cfg.device)
    loss_func = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


    for dir_ in cfg.dirs_logs:
        if dir_ not in os.listdir(cfg.path_to_logs):
            os.makedirs(os.path.join(cfg.path_to_logs, dir_))


    train(model,loss_func,optimizer,cfg)
    model, optimizer, start_epoch, max_accuracy = load_model(os.path.join(cfg.path_to_saves, cfg.name_save),
                                                             model,
                                                             cfg,
                                                             optimizer)

    test(model, cfg, start_epoch)


if __name__ == '__main__':
    main()














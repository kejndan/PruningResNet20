import torch
import clearml
import os


def to_logger(path,step,val):
    folder = path[:path.find('/')]
    file = path[path.find('/')+1:]
    clearml.Logger.current_logger().report_scalar(folder, file, iteration=step, value=val)

def log_metric(path, step, metrics,cfg):
    with open(os.path.join(cfg.path_to_logs, path), 'a') as f:
        f.write(f'{step} {metrics} \n')

def load_model(load_path, model, cfg, optimizer=None):
    checkpoint = torch.load(load_path)
    load_state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if key in load_state_dict:
            model_state_dict[key] = load_state_dict[key]
    model.load_state_dict(model_state_dict)
    start_epoch = checkpoint['epoch'] + 1
    max_accuracy = checkpoint['best_accuracy']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['opt'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.device)
    return model, optimizer, start_epoch, max_accuracy
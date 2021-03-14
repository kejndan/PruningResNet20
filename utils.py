import torch
import clearml
import os
import matplotlib.pyplot as plt


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

def plot(name,axes, cfg,xlabel=None,ylabel=None,plot_label=None):
    x = []
    y = []
    with open(os.path.join(cfg.path_to_logs,name),'r') as f:
        for line in f.readlines():
            v1, v2 = line.split()
            x.append(int(v1))
            y.append(float(v2))
    if plot_label is not None:
        axes.plot(x,y,labels=plot_label)
    else:
        axes.plot(x,y)
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    axes.set_title(name)

def plot_F_with_axes(axes, store):
    names = list(store.keys())
    idx = 0
    for row in range(6):
        for column in range(5):
            name = names[idx]
            length = len(store[name][0])
            accuracy = np.array(store[name][1])
            uniq_params = np.array(store[name][0])
            axes[row, column].plot(np.arange(2,length+2),accuracy/uniq_params)
            axes[row,column].set_xlabel('Nums clusters')
            axes[row,column].set_ylabel('F')
            axes[row,column].set_title(name)
            idx += 1
            if idx == len(names):
                break
        if idx == len(names):
            break
    plt.show()
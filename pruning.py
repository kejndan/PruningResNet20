
import numpy as np
import torch
import os
from model import ResNet20
from config import cfg
from utils import load_model
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from test import test

def check_name(name, names_layers):
    if names_layers is not None:
        if isinstance(names_layers, str) and name == names_layers\
                or isinstance(names_layers,list) and name in names_layers:
            return True
        else:
            return False
    else:
        return True

def clustering_filters_models(model, clustering_algorithm, init_algorithm=None, nums_clusters=None, names_layers=None):
    store_centers = {}
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, torch.nn.Conv2d) and check_name(name, names_layers):
            nb_clusters = 16
            if nums_clusters is not None:
                for name_layer, nb in nums_clusters.items():
                    if name_layer in name:
                        nb_clusters = nb

            centers, clusters = clustering_layer(module, nb_clusters,
                                                 clustering_algorithm,
                                                 init_algorithm=init_algorithm)
            store_centers[name] = [np.array(centers),
                                   clusters]
    return store_centers

def clustering_layer(module, nb_clusters,clustering_algorithm, init_algorithm=None):
    filters = module.weight.data
    vectors = filters.view(filters.size(0), -1).cpu().numpy()
    if init_algorithm is not None:
        initial_centers = init_algorithm(vectors, nb_clusters).initialize()
    clust_func = clustering_algorithm(vectors, initial_centers)
    clust_func.process()
    return clust_func.get_centers(), clust_func.get_clusters()

def pruning_model(model, store_centers, cfg,names_layers=None):

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d) and check_name(name, names_layers):
            centers = store_centers[name]
            tensor_centers = []
            for center in centers[0]:
                tensor_centers.append(torch.Tensor(center.reshape(module.weight[0].squeeze(0).size())).to(cfg.device))
            for idx, cluster in enumerate(centers[1]):
                module.weight.data[cluster] = tensor_centers[idx]

if __name__ == '__main__':
    model = ResNet20().to(cfg.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    for dir_ in cfg.dirs_logs:
        if dir_ not in os.listdir(cfg.path_to_logs):
            os.makedirs(os.path.join(cfg.path_to_logs, dir_))
    cfg.logger = None
    # train(model,loss_func,optimizer,cfg)
    model, optimizer, start_epoch, max_accuracy = load_model(os.path.join(cfg.path_to_saves, cfg.name_save),
                                                             model,
                                                             cfg,
                                                             optimizer)
    # test(model, cfg, start_epoch)
    nb_clusters = {'layer1': 16, 'layer2': 32, 'layer3': 64}
    clustering_filters = clustering_filters_models(model, kmeans, kmeans_plusplus_initializer,nums_clusters=nb_clusters)
    pruning_model(model,clustering_filters,cfg)
    test(model, cfg, start_epoch)



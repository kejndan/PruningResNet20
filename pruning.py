import numpy as np
import torch
from config import cfg
from copy import deepcopy
import re


def clustering_layer(module, nb_clusters,clustering_algorithm, init_algorithm):
    """
    Данная функция кластеризует сверточные фильтры у переданного сверточного слоя.
    :param module: сверточный слой
    :param nb_clusters: количество кластеров
    :param clustering_algorithm: алгоритм кластеризации
    :param init_algorithm: алгоритм инициализации центройд
    :return: центроды кластеров, списки с тем к каким кластерам были отнесены сверточные фильтры
    """
    filters = module.weight.data
    vectors = filters.view(filters.size(0), -1).cpu().numpy()

    initial_centers = init_algorithm(vectors, nb_clusters, random_state=cfg.random_seed).initialize()
    clust_func = clustering_algorithm(vectors, initial_centers)
    clust_func.process()
    return clust_func.get_centers(), clust_func.get_clusters()

def pruning_model(model, store_centers, cfg):
    """
    Данная функция заменяет сверточные фильтры на центройды кластеров, к которым они относятся.
    :param model: переданная модель
    :param store_centers: dict(key - название слоя, value - [центройды, метки сверток])
    :return: модель после прунинга
    """
    new_model = deepcopy(model)
    for name, module in new_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name in store_centers:
            centers = store_centers[name]
            tensor_centers = []

            # Представление центройд в виде тензоров
            for center in centers[0]:
                tensor_centers.append(torch.Tensor(center.reshape(module.weight[0].squeeze(0).size())).to(cfg.device))

            for idx, cluster in enumerate(centers[1]):
                module.weight.data[cluster] = tensor_centers[idx]

    return new_model


def prune_layers_by_regexp(expression,nb_clusters,model,cfg, clust_func, init_func):
    """
    Данная функция является оболочкой для прунинга модели по названиям слоёв, которые удовлетворяют регулярному выражению
    :param expression: регулярное выражение, по которому будет отбраны слои
    :param nb_clusters: количество кластеров
    :param model: модель
    :param clust_func: функция кластеризации
    :param init_func: алгоритм инициализации центройд
    """
    return prune_this_layers(expression, nb_clusters, model, cfg, clust_func, init_func)

def prune_layers_by_names(names,model,cfg, clust_func, init_func):
    """
    Данная функция является оболочкой для прунинга модели по названием слоёв
    :param expression: dict(key - название слоя, value - количество кластеров для данного слоя)
    :param model: модель
    :param clust_func: функция кластеризации
    :param init_func: алгоритм инициализации центройд
    """
    return prune_this_layers(names, None, model, cfg, clust_func, init_func)

def prune_this_layers(condition,nb_clusters,model, cfg, clust_func, init_func):
    """
    Данная функция выполняет прунинг модели для тех слоёв название, которых удовлетворяет condition.
    :param condition: regexp или dict
    :param nb_clusters: количество кластеров. !если type(condition) == dict, то количество кластеров берется из value.
    :param model: модель
    :param cfg:
    :param clust_func: функция кластеризации
    :param init_func: алгоритм инициализации центройд
    """

    if isinstance(condition, str):
        prune_by = 'regexp'
    elif isinstance(condition, dict):
        prune_by = 'names'
    store_centers = {}
    count_changed_layers = 0
    unique_params = 0
    conv_params = 0
    for name, module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):

            if prune_by == 'regexp' and re.match(condition, name):
                nb_clusters = nb_clusters
            elif prune_by == 'names' and name in condition:
                nb_clusters = condition[name]
            else:
                continue

            count_changed_layers += 1
            centers, clusters = clustering_layer(module,nb_clusters,clust_func,init_func)
            centers = np.array(centers)
            store_centers[name] = [centers,clusters]
            unique_params += len(np.unique(centers))
            conv_params += np.prod(module.weight.size())
    pruned_model = pruning_model(model,store_centers,cfg)
    return pruned_model,count_changed_layers, unique_params, conv_params

if __name__ == '__main__':
    print('Pruning module')




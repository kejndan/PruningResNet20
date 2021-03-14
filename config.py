from easydict import EasyDict

cfg = EasyDict()
cfg.random_seed = 0
cfg.device = 'cuda:0'

cfg.path_to_dataset = 'C:\\Users\\adels\PycharmProjects\datasets\cifar10'
cfg.path_to_saves = 'saves'
cfg.path_to_logs = 'logs'
cfg.dirs_logs = ['eval','loss']

cfg.clearml_logging = False
cfg.path_to_keys = 'keys.txt'
cfg.project_name = 'Demo'
cfg.task_name = 'Find best score'
cfg.logger = None

cfg.evaluate_on_train_data = True
cfg.evaluate_on_validation_data = False
cfg.valid_size = 0.2

cfg.load_save = False
cfg.name_save = 'best_checkpoint'


cfg.batch_size = 128
cfg.lr = 1e-1
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.total_epochs = 100


cfg.save_models = True
cfg.save_best_model = True


cfg.use_lr_scheduler = True
cfg.ROP_patience = 5
cfg.ROP_factor = 0.1

cfg.plot_confusion_matrix = True






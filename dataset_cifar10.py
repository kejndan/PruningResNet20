import torch
from torchvision import transforms
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter
import torchvision


def unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, path, cfg, work_mode='train', transform_mode='train', augmentation=True):
        self.path_dir = path
        self.work_mode = work_mode
        self.transform_mode = transform_mode
        self.augmentation = augmentation
        self.train_paths = [os.path.join(self.path_dir,f'data_batch_{i}') for i in range(1, 6)]
        self.test_paths = os.path.join(self.path_dir,'test_batch')
        self.meta_path = os.path.join(self.path_dir,'batches.meta')
        self.crop_size = 32
        self.angle_rotation = 30
        self.means = np.array((0.4914, 0.4822, 0.4465))
        self.stds = np.array((0.247, 0.243, 0.262))

        train_batches = list(map(unpickle, self.train_paths))
        test_batches = unpickle(self.test_paths)
        self.name_classes = [name.decode('utf-8') for name in unpickle(self.meta_path)[b'label_names']]
        self.nb_classes = 10

        if work_mode == 'train' or work_mode == 'valid':
            full_train_X = np.concatenate([batch[b'data'] for batch in train_batches]).reshape([-1, 3, 32, 32]).astype('uint8')
            full_train_y = np.concatenate([batch[b'labels'] for batch in train_batches]).astype('int32')
            full_train_X = full_train_X.transpose((0, 2, 3, 1))

            if work_mode == 'train':
                if cfg.evaluate_on_validation_data:
                    self.X, _, self.y, _ = train_test_split(full_train_X, full_train_y, test_size=cfg.valid_size,
                                                            random_state=cfg.random_seed)
                    print(f'train {Counter(self.y)}')
                else:
                    self.X, self.y = full_train_X, full_train_y

            elif work_mode == 'valid':
                if cfg.evaluate_on_validation_data:
                    _, self.X, _, self.y = train_test_split(full_train_X, full_train_y, test_size=cfg.valid_size,
                                                            random_state=cfg.random_seed)
                else:
                    self.X, self.y = None, None
                print(f'valid {Counter(self.y)}')

        elif work_mode == 'test':
            self.X = test_batches[b'data'].reshape([-1, 3, 32, 32]).astype('uint8')
            self.X = self.X.transpose((0, 2, 3, 1))

            self.y = np.array(test_batches[b'labels']).astype('int32')

    def apply_augmentation(self, img):
        img = Image.fromarray(img)
        if self.transform_mode == 'train':
            transforms_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ])
        else:
            transforms_aug = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ])
        return transforms_aug(img)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.augmentation:
            x = self.apply_augmentation(self.X[idx])
        else:
            x = self.X[idx]
        return x, self.y[idx], idx


if __name__ == '__main__':
    print('Dataset CIFAR10 module')
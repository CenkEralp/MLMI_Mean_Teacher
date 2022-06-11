from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy

import medmnist
from medmnist import INFO, Evaluator

import os
from torchvision.utils import save_image
import matplotlib.image

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
lr = 0.001
weight_decay = 0.001
number_of_res_blocks = 5

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
task

workdir = os.getcwd() + "/data-local/images/"
os.makedirs(workdir, exist_ok=True)

train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)


train_dir = os.path.abspath(os.path.join(workdir, 'train'))
val_dir = os.path.abspath(os.path.join(workdir, 'train'))
test_dir = os.path.abspath(os.path.join(workdir, 'test'))

number_of_labeled = 10000
labeled_indeces = np.random.uniform(low=0, high=len(train_dataset)-1, size=(number_of_labeled,)).astype('int')
unlabeled_indeces = np.delete(range(len(train_dataset)), labeled_indeces)

label_names = info['label']
print(label_names)

def write_image(target_dir, index, x, y):
    subdir = os.path.join(target_dir, str(y))
    name = "{}_{}.png".format(index, y)
    os.makedirs(subdir, exist_ok=True)
    save_image(x, os.path.join(subdir, name))
    return

file_object = open(os.path.join(workdir, "labels.txt"), 'a')

for i in range(len(train_dataset)):
    x, y = train_dataset[i][0], train_dataset[i][1]
    
    write_image(train_dir, i, x, y)
    if i in labeled_indeces:
        file_object.write('{}_{}.png {}'.format(i, y, y))

file_object.close()

for i in range(len(val_dataset)):
    x, y = val_dataset[i][0], val_dataset[i][1]
    write_image(val_dir, i, x, y)

for i in range(len(test_dataset)):
    x, y = test_dataset[i][0], test_dataset[i][1]
    write_image(test_dir, i, x, y)    
    
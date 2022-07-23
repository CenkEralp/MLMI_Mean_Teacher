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
import random
import sys

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
val_dir = os.path.abspath(os.path.join(workdir, 'val'))
test_dir = os.path.abspath(os.path.join(workdir, 'test'))

number_of_labeled = int(sys.argv[1])
print("----------number_of_labeled:", number_of_labeled)
number_of_class = 9
"""
labeled_indeces = [i for i in range(len(train_dataset))]
random.shuffle(labeled_indeces)
labeled_indeces = labeled_indeces[0:number_of_labeled]
unlabeled_indeces = np.delete(range(len(train_dataset)), labeled_indeces)
"""
unbalanced = True
small_validation = False

Train_len = len(train_dataset)
labels = np.array([train_dataset[i][1][0] for i in range(Train_len)])

label_per_class = number_of_labeled // number_of_class
labeled_indeces = []
unlabeled_idx = np.array(range(len(labels))) 
for i in range(number_of_class): 
    idx = np.where(labels == i)[0] 
    idx = np.random.choice(idx, label_per_class, False) 
    labeled_indeces.append(idx) 
labeled_indeces = np.array(labeled_indeces)

if unbalanced:
    unlabeled_indeces = []
    unlabeled_classes = [6, 7, 8]
    for i in unlabeled_classes:
        idx = np.where(labels == i)[0] 
        #idx = np.random.choice(idx, len(labels) // len(unlabeled_classes), False) 
        unlabeled_indeces.append(idx)
else:
    unlabeled_indeces = range(Train_len)

unlabeled_indeces = np.array(unlabeled_indeces)

label_names = info['label']
print(label_names)

def write_image(target_dir, index, x, y):
    subdir = os.path.join(target_dir, str(y[0]))
    name = "{}_{}.png".format(index, y[0])
    os.makedirs(subdir, exist_ok=True)
    t = transforms.ToTensor()
    save_image(t(x), os.path.join(subdir, name))
    return

file_object = open(os.path.join(workdir, "labels.txt"), 'a')

for i in range(len(train_dataset)):
    x, y = train_dataset[i][0], train_dataset[i][1]
    
    write_image(train_dir, i, x, y)
    if i in labeled_indeces:
        file_object.write('{}_{}.png {}\n'.format(i, y[0], y[0]))

file_object.close()

len_val_dataset = len(val_dataset) if small_validation else int(len(val_dataset) * 0.2)

for i in range(len_val_dataset):
    x, y = val_dataset[i][0], val_dataset[i][1]
    write_image(val_dir, i, x, y)

for i in range(len(test_dataset)):
    x, y = test_dataset[i][0], test_dataset[i][1]
    write_image(test_dir, i, x, y)    
    
﻿from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pylot as plt

import os 
import time

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val' : transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x]) 
                 for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4,
                                              shuffle=True, num_workers = 4)
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x])
                for x in ['train', 'val']}

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_avaliable()

def imshow(inp, title=None) :
    """Imshow for Tensor."""
    inp.numpy().transpose((1, 2, 0))

inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25) :
    """train the model"""
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs) :
        print('Epoch {}/{}}', format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val'] :
            if phase == 'train' :
                scheduler.step()
                model.train(True)
            else :
                model.train(False)

        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders[phase] :
            inputs, labels = data
        
            if use_gpu :
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else :
                inputs = Variable(inputs)
                labels = Variable(labels)
           
            optimizer.zero_grad()

            #forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
  
            if phase == 'train' :
                loss.backward()
                optimizer.step()

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds = labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]
 
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        print()
model_ft = models.resnet18(pretrained = True)
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features, 2)
if use_gpu :
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)
exp_lr_scheduler = lr_schedulers.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 25)
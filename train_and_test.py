# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import numpy as np
from stratified_group_data_splitting import StratifiedGroupKFold
from create_dataset import ProstateDataset, ToTensorDataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.model_t2 import Net
from utils import EarlyStopping
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, recall_score, precision_score
import statistics as stat
import random

# Configuration
num_epochs = 100
batch_size_train = 4
batch_size_test = 1
patience = 3 
n_splits = 5
n_images = 15
N = 10  # Repetitions of 5-fold CV
csv_file = ""
output_path = ""
dataset = ProstateDataset(csv_file)
aug_suffix = ['rotation', 'vertical_flip', 'horizontal_flip', 'translation']
allsets_aug = [ProstateDataset(f"{suffix}.csv") for suffix in aug_suffix]

# Prepare datasets
X = np.array([dataset[i][0] for i in range(len(dataset))])
y = np.array([dataset[i][1].item() for i in range(len(dataset))])
patients = np.array([dataset[i][2] for i in range(len(dataset))])

# Evaluation metrics
results = {}
metrics = ['Specificity', 'Sensitivity', 'Balanced accuracy', 'Precision', 'AUC']
mean_of_means = {metric: [] for metric in metrics}

# Loss function and stopping criterion
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=patience, verbose=True)

# Training
for repetition in range(N):
    print(f"*** Repetition: {repetition}")

    cv = StratifiedGroupKFold(n_splits=n_splits, random_state=repetition, shuffle=True)
    results[f"Repetition {repetition}"] = []

    for outer_iteration, (train_val_idxs, test_idxs) in enumerate(cv.split(X, y, patients), start=1):
        print(f"Train n°: {outer_iteration}, Repetition n°: {repetition}")

        train_valset = [dataset[i] for i in train_val_idxs]
        testset = [dataset[i] for i in test_idxs]

        X_tv = [train_valset[i][0] for i in range(len(train_valset))]
        y_tv = [train_valset[i][1] for i in range(len(train_valset))]

        train_idxs, val_idxs = train_test_split(np.arange(len(y_tv)), test_size=0.10, stratify=y_tv, random_state=1)
        trainset = [train_valset[i] for i in train_idxs]
        validset = [train_valset[i] for i in val_idxs]

        for allset_aug in allsets_aug:
            train_valset_aug = [allset_aug[i] for i in train_val_idxs]
            trainset_aug = [train_valset_aug[i] for i in train_idxs]
            hg_positions = [i for i in range(len(trainset_aug)) if trainset_aug[i][1].item() == 1]
            positions = random.sample(hg_positions, n_images)
            trainset.extend(trainset_aug[position] for position in positions)

        to_tensor = transforms.ToTensor()
        trainset_tf = ToTensorDataset(trainset, to_tensor)
        validset_tf = ToTensorDataset(validset, to_tensor)
        testset_tf = ToTensorDataset(testset, to_tensor)

        trainloader = DataLoader(trainset_tf, batch_size=batch_size_train, shuffle=True)
        validationloader = DataLoader(validset_tf, batch_size=batch_size_train, shuffle=False)
        testloader = DataLoader(testset_tf, batch_size=batch_size_test, shuffle=False)

        dset_loaders = {'train': trainloader, 'val': validationloader}
        classes = ('LG', 'HG')

        net = Net()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}\n{"-" * 10}')

            for phase in ['train', 'val']:
                net.train(phase == 'train')

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dset_loaders[phase]:
                    inputs, labels = inputs.float().to(device), labels.long().to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    running_corrects += (predicted == labels).sum().item()

                epoch_loss = running_loss / len(dset_loaders[phase])
                epoch_acc = running_corrects / len(dset_loaders[phase])
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss = epoch_loss
                else:
                    valid_loss = epoch_loss

            early_stopping(valid_loss, train_loss)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('Finished Training\n')

        torch.save(net.state_dict(), output_path)

        net.eval()
        net.to("cpu")

        correct, total = 0, 0
        actuals, predictions, class_probabilities = [], [], []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.float()
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item())

                actuals.extend(labels.view_as(predicted) == 0)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                class_probabilities.extend(probabilities[:, 0])

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        actuals = [i.item() for i in actuals]
        true_labels = [0 if i else 1 for i in actuals]
        class_probabilities = [i.item() for i in class_probabilities]

        fpr, tpr, _ = roc_curve(actuals, class_probabilities)

        cv_results_sk = {
            'Specificity': recall_score(true_labels, predictions, pos_label=0),
            'Sensitivity': recall_score(true_labels, predictions),
            'Balanced accuracy': balanced_accuracy_score(true_labels, predictions),
            'Precision': precision_score(true_labels, predictions),
            'AUC': auc(fpr, tpr)
        }

        results[f'Repetition {repetition}'].append(cv_results_sk)

    mean_results = {metric: stat.mean([results[f'Repetition {repetition}'][i][metric] for i in range(n_splits)]) for metric in metrics}
    results[f'Repetition {repetition}'].append(mean_results)

    for metric in metrics:
        mean_of_means[metric].append(mean_results[metric])

for metric in metrics:
    print(metric, stat.mean(mean_of_means[metric]))

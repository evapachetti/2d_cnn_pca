# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from models.model_t2 import Net
from create_dataset import ProstateDataset, ToTensorDataset

input_csv = ""
Nrep = 1000

def performance_bootstrap(input_csv, Nrep):
    data = pd.read_csv(input_csv)
    keys = data['patient_id'].tolist()
    les_gt = data['label'].tolist()
    lesions = data['lesion_id'].unique()

    bootstrap_sample_size = len(lesions)
    test_AUROC = []

    for _ in range(Nrep):
        les_gt_boot = []
        bootstrap_sample = np.random.choice(lesions, bootstrap_sample_size, replace=True)

        new_indices = data[data['lesion_id'].isin(bootstrap_sample)].index.tolist()
        ind_data_boot = data.loc[new_indices]

        for lesion in bootstrap_sample:
            idx = keys.index(lesion)
            les_gt_boot.append((lesion, les_gt[idx]))

        auroc = performance_lesion_level(ind_data_boot, les_gt_boot)
        test_AUROC.append(auroc)

    return test_AUROC

def performance_lesion_level(ind_data_boot, les_gt_boot):
    testset = ProstateDataset(ind_data_boot)
    testset_tf = ToTensorDataset(testset, transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset_tf, batch_size=1, shuffle=False, num_workers=0)

    net = Net()
    net.eval()

    device = torch.device("cpu")
    net.to(device)

    PATH = ""
    net.load_state_dict(torch.load(PATH))

    slice_probs = {lesion: [] for lesion, _ in les_gt_boot}
    lesion_probabilities = []

    with torch.no_grad():
        for data in testloader:
            image, _, _, lesion, _ = data[0], data[1], data[3], data[4]
            image = image.float()
            output = net(image)

            p = torch.nn.functional.softmax(output, dim=1)
            slice_probs[lesion[0].item()].append(p[0][1].item())

    for lesion in slice_probs:
        mean_prob_1 = np.mean(slice_probs[lesion])
        lesion_probabilities.append(mean_prob_1)

    true_lesions = [gt[1] for gt in les_gt_boot]

    auroc = roc_auc_score(true_lesions, lesion_probabilities)

    return auroc

test_AUROC = performance_bootstrap(input_csv, Nrep)
test_AUROC = [auroc for auroc in test_AUROC if not np.isnan(auroc)]

median = np.median(test_AUROC)
perc_1 = np.percentile(test_AUROC, 5)
perc_2 = np.percentile(test_AUROC, 95)

print(f"Median AUROC: {median}")
print(f"5th Percentile: {perc_1}")
print(f"95th Percentile: {perc_2}")

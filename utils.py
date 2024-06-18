import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import MetricCollection, Accuracy, AUROC
from torchmetrics import F1Score as F1

def seed_everything(seed=42):
    pl.seed_everything(seed) # ADDED
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_parameters(args, save_dir):
    print(args)
    args_dict = vars(args)
    with open(os.path.join(save_dir, "parameters.json"), "w") as f:
        json.dump(
            {n: str(args_dict[n]) for n in args_dict},
            f,
            indent=4
        )

class switch_dim(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.transpose(x, 2, 1)
        return x

def get_metrics(num_classes, task):
    metrics = MetricCollection({
        "ACC": Accuracy(num_classes=num_classes, task=task), # default: average="micro" -> might be misleading if classes are imbalanced
        # "Balanced_ACC": Accuracy(num_classes=num_classes, average="macro", task=task), # does not take label imbalance -> helping in situations where each class's prediction is equally important.
        "AUROC": AUROC(num_classes=num_classes, task=task),
        "F1": F1(num_classes=num_classes, task=task)
    })

    return metrics

def get_loss_weight(args, data_module):
    if args.loss_weight is not None:
        loss_weight = args.loss_weight
    elif args.auto_loss_weight: # automatically calculate the weight of each class
        data_module.setup()
        loss_weight = data_module.dataset_train.get_weight_of_class()
    else:
        loss_weight = None
    if loss_weight is not None:
        print("Using loss weight:", loss_weight)
        loss_weight = torch.tensor(loss_weight)

    return loss_weight

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


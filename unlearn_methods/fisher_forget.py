
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn import linear_model, model_selection
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn import linear_model, model_selection
import torchvision.models as models
from sklearn.cluster import KMeans
from utils.utils import *
from tqdm import tqdm
from copy import deepcopy



def hessian(dataset, model, device, num_classes):
    """
    Approximates the Hessian (second-order info) for each parameter of 'model'
    on 'dataset'. Accumulates in p.grad2_acc.
    
    Arguments:
        dataset: PyTorch Dataset
        model:   Neural network model
        device:  Device to perform computations on
        num_classes: Number of classes in the classification problem
    """
    model.eval()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize second-order accumulations
    for p in model.parameters():
        p.grad2_acc = torch.zeros_like(p.data)
    
    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        # Detach softmax probabilities for weighting
        prob = F.softmax(output, dim=-1).detach()

        for y in range(num_classes):
            target = torch.full_like(orig_target, y)  # Construct a label 'y'
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    # If batch_size=1, .mean() is just prob[0, y], but this is robust to larger batches as well
                    p.grad2_acc += prob[:, y].mean() * p.grad.data.pow(2)
    
    # Normalize by the number of mini-batches
    for p in model.parameters():
        p.grad2_acc /= len(train_loader)


def get_mean_var(p, is_base_dist=False, alpha=3e-6, num_classes=None, class_to_forget=None):
    """
    Computes the mean (mu) and variance (var) for the Gaussian noise injection
    based on the approximated Hessian in p.grad2_acc. 
    
    Arguments:
        p:            A parameter tensor from the model
        is_base_dist: (Unused in this example) Whether to use the base distribution
        alpha:        Scale factor for variance
        num_classes:  Number of classes (to identify last layer)
        class_to_forget: If not None, sets mu/var for that class to "forget" it
    """
    # 1. Invert Hessian-based second-order stats -> var ~ alpha / (grad2_acc + 1e-8)
    var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
    
    # 2. Clamp to avoid exploding
    var = var.clamp(max=1e3)
    if num_classes is not None and p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    
    # 3. Scale by alpha
    var = alpha * var

    # 4. For multi-dimensional params (e.g. convolutional filters), optionally average across dim=1
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    
    # 5. Base mean is the original parameter value p.data0
    mu = deepcopy(p.data0)

    # 6. (Optional) If we have a class to forget and this is the last layer
    #    zero out that row's mean and set a small variance
    if class_to_forget is not None and num_classes is not None and p.size(0) == num_classes:
        mu[class_to_forget] = 0.0
        var[class_to_forget] = 1e-4  # small variance
    
    # 7. Scale variance for last layer or BatchNorm parameters
    if num_classes is not None and p.size(0) == num_classes:
        var *= 10
    elif p.ndim == 1:
        # e.g. BatchNorm bias/gamma/beta can be 1D
        var *= 10

    return mu, var


def fisher_forgetting(model, retain_loader, num_classes, device, 
                      class_to_forget=None, num_to_forget=None, alpha=1e-6):
    """
    Applies a "fisher forgetting" step on the model by:
    1) Storing the old parameters (p.data0).
    2) Computing Hessian approx (p.grad2_acc) via 'hessian(...)'.
    3) Sampling new parameters from N(mu, var) where mu, var come from 'get_mean_var()'.
    
    Arguments:
        model:            The model to be modified
        retain_loader:    A DataLoader whose dataset is used for Hessian approximation
        num_classes:      Number of classes
        device:           Device for computation
        class_to_forget:  If not None, forcibly zero out that class's row in final layer
        num_to_forget:    (Unused here, but kept for signature compatibility)
        alpha:            Scale factor for variance
    """
    # Store original parameters before noise injection
    for p in model.parameters():
        p.data0 = deepcopy(p.data)

    # Compute Hessian approximation
    hessian(retain_loader.dataset, model, device, num_classes)

    # Inject noise according to the Hessian-based Gaussian
    for p in model.parameters():
        mu, var = get_mean_var(
            p, 
            is_base_dist=False, 
            alpha=alpha, 
            num_classes=num_classes, 
            class_to_forget=class_to_forget
        )
        # Sample noise from Normal(0, 1), scale by sqrt(var), add to mu
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

    return model
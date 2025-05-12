import warnings
warnings.filterwarnings('ignore')
import torch
from utils.utils import *
from torch.utils.data import DataLoader


def extract_features(model, dataloader, device):
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            # feature = model(data)
            feature = model.embed(data)
            feature = feature.view(feature.size(0), -1)  # Flatten spatial dimensions
            features.append(feature.cpu())
            labels.append(label)
    
    return torch.cat(features, 0), torch.cat(labels, 0)




def kmeans_pytorch(X, num_clusters, num_iterations=100, tol=1e-4):
    N, D = X.shape
    
    # Randomly initialize cluster centers
    C = X[torch.randperm(N)[:num_clusters]]
    
    for i in range(num_iterations):
        # Compute distances
        distances = torch.cdist(X, C)
        
        # Assign points to nearest cluster
        labels = torch.argmin(distances, dim=1)
        
        # Update cluster centers
        new_C = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])
        
        # Check for convergence
        if torch.abs(new_C - C).sum() < tol:
            break
        
        C = new_C
    
    return labels
    



def create_sub_classes(inputs, 
                       labels, 
                       model, 
                       num_classes=10, 
                       sub_divisions=10,
                       device=torch.device('cuda')):


    new_labels = torch.zeros_like(labels)
    
    indices_train_wrt_finelabels = {}
    
    model.to(device)

    dataset = TensorDatasett(inputs, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    features_all, _ = extract_features(model, loader, device)
    
    for i in range(num_classes):
        mask = (labels == i)
        class_features = features_all[mask]
        
        
        class_new_labels = kmeans_pytorch(
            class_features, 
            num_clusters=sub_divisions
        )
        
        # Create sublabels
        new_subclass_labels = i * sub_divisions + class_new_labels
        
        # Assign new labels
        new_labels[mask] = new_subclass_labels


    new_labels_unique = torch.unique(new_labels).tolist()
    
    subclass_info = {}  # Will store centroid, radius, epsilon for each fine label
    
    for fine_label in new_labels_unique:
        # Indices of data belonging to this fine-label
        idxs = torch.where(new_labels == fine_label)[0]
        indices_train_wrt_finelabels[fine_label] = idxs.tolist()
        
        # Ref: I call sub-class or fine label or clusters the same thing, and call features as points
        #  Compute centroid, radius, epsilon in feature space
        #  - centroid is the mean of each cluster features
        #  - radius is max distance from centroid to any known point in cluster
        #  - epsilon is the minimum pairwise distance among points in that sub-class
        #  - I assign the epislon as 1/3 of minimum distance among points in that sub-class
        
        feats = features_all[idxs]  # shape: (n, D)
        if len(feats) == 0:
            subclass_info[fine_label] = {
                'centroid': None,
                'radius': 0.0,
                'epsilon': 0.0
            }
            continue
        
        centroid = feats.mean(dim=0)   # shape (feature)
        
        dists = torch.norm(feats - centroid, dim=1)   # shape (num points)
        radius = dists.max().item()   # scaler
        

        if len(feats) == 1:
            epsilon = 1e-5
        else:
            dist_mat = torch.cdist(feats, feats, p=2)  # shape (n, n)
            # Exclude diagonal by setting it to a large number or filtering
            dist_mat.fill_diagonal_(float('inf'))
            epsilon = dist_mat.min().item()
            epsilon = epsilon/3
        
        subclass_info[fine_label] = {
            'centroid': centroid,
            'radius': radius,
            'epsilon': epsilon
        }
    
    return new_labels, indices_train_wrt_finelabels, subclass_info


def dataset_sampling(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels,
                     ):
    


    
    target_images = []
    target_labels = []
    
    # Process each array in indices_train_wrt_finelabels
    for i, (_,arr) in enumerate(indices_train_wrt_finelabels.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        

        target_images.append(train_images[list(arr)])
        target_labels.append(train_labels[list(arr)])


    
    return target_images, target_labels





def seperated_dataset_sampling(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels,
                     forget_indices
                     ):
    
    F_set = set(forget_indices)


    target_images = []
    target_labels = []

    residual_indices = []

    # Process each array in indices_train_wrt_finelabels
    for i, (_,arr) in enumerate(indices_train_wrt_finelabels.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        
        if not F_set.intersection(arr_set):
            target_images.append(train_images[list(arr)])
            target_labels.append(train_labels[list(arr)])

        else:
            res_arr = arr_set - F_set
            residual_indices.extend(res_arr)

    residual_indices = list(residual_indices)
    residual_images = train_images[residual_indices]
    residual_labels = train_labels[residual_indices]

    return target_images, target_labels, residual_images, residual_labels







def condensation_sampling(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels
                     ):
    
    
    target_images = []
    target_labels = []
    sub_labels = []

    # Process each array in indices_train_wrt_finelabels
    for i, (sub_lab,arr) in enumerate(indices_train_wrt_finelabels.items()):

        target_images.append(train_images[list(arr)])
        target_labels.append(train_labels[list(arr)])
        sub_labels.append(sub_lab)


    return target_images, target_labels, sub_labels










def seperated_sampling_v2(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels,
                     forget_indices,
                     sub_labels,
                     condensed_images,
                     condensed_labels
                     ):
    
    F_set = set(forget_indices)

    residual_indices = []
    cond_imgs = []
    cond_labels = []

    # Process each array in indices_train_wrt_finelabels
    for i, (sub_lab,arr) in enumerate(indices_train_wrt_finelabels.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        
        if F_set.intersection(arr_set):
            res_arr = arr_set - F_set
            residual_indices.extend(res_arr)

        else:
            idx = sub_labels.index(sub_lab)
            cond_imgs.append(condensed_images[idx].unsqueeze(0))
            cond_labels.append(condensed_labels[idx].unsqueeze(0))

    residual_indices = list(residual_indices)
    residual_images = train_images[residual_indices]
    residual_labels = train_labels[residual_indices]

    cond_images = torch.cat(cond_imgs, dim=0)
    cond_labels = torch.cat(cond_labels, dim=0)

    return cond_images, cond_labels, residual_images, residual_labels



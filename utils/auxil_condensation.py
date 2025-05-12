import warnings
warnings.filterwarnings('ignore')
import torch




def sample_images(images, labels, target_label, N=100):
    indices = torch.where(labels == target_label)[0]
    sampled_indices = torch.randperm(len(indices))[:N]
    return images[sampled_indices]



def noisy_augment(normalized_img, K=5):

    if K <= 0:
        # Return the original image
        return normalized_img

    # Ensure the input has the correct shape
    if normalized_img.dim() != 4 or normalized_img.size(0) != 1:
        raise ValueError(f"Expected input tensor of shape [1, C, H, W], but got {normalized_img.shape}")

    # Repeat the image K times to create a batch of noisy images
    # Shape after repeat: [K, C, H, W]
    normalized_img_batch = normalized_img.repeat(K, 1, 1, 1)

    # Generate K noise means and standard deviations
    # Shape: [K, 1, 1, 1] to broadcast across [C, H, W]
    noise_mean = torch.empty(K, 1, 1, 1, device=normalized_img.device).uniform_(0, 0.1)
    noise_std = torch.empty(K, 1, 1, 1, device=normalized_img.device).uniform_(0, 0.1)

    # Generate noise: mean + std * random noise
    noise = noise_mean + noise_std * torch.randn_like(normalized_img_batch)

    # Add noise to the images
    noisy_imgs = normalized_img_batch + noise

    # Clamp the noisy images to be within [0, 1]
    noisy_imgs = torch.clamp(noisy_imgs, 0, 1)

    # Concatenate the original image with the noisy images
    # Shape: [1 + K, C, H, W]
    augmented_imgs = torch.cat([normalized_img, noisy_imgs], dim=0)

    return augmented_imgs


def get_data_to_be_condensed(fine_lab_idx_dictionary, train_images, train_labels, forget_indices, device):
        # Convert forget_indices to a set for O(1) lookup
    F_set = set(forget_indices)

    target_images = []
    target_labels = []

    # Process each array in indices_train_wrt_finelabels
    for i, (key, arr) in enumerate(fine_lab_idx_dictionary.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        
        if F_set.intersection(arr_set):
            # If any elements of F are in the array, add all non-F elements to residual_indices
            res_indices=list(arr_set - F_set)
            target_images.append(train_images[res_indices].to(device))
            target_labels.append(train_labels[res_indices].to(device))

    return target_images, target_labels



def check_retain_size(retain_labels, forget_labels, retain_images, train_images, train_labels, img_batch_size):
    # Get unique forget labels
    unique_forget = torch.unique(forget_labels)
    
    # Check counts for each forget label in retain set
    for f_label in unique_forget:
        retain_count = torch.sum(retain_labels == f_label).item()
        if retain_count < img_batch_size:
            # If any class has fewer samples than needed, return full training set
            return train_images, train_labels
            
    # All classes have sufficient samples, return retain set
    return retain_images, retain_labels
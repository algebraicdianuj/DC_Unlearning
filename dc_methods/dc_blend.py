import warnings
warnings.filterwarnings('ignore')
import torch
from utils.utils import *
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from utils.dc_losses import *



def denormalize(tensor, mean, std):
    # The denormalization formula is: denormalized = normalized * std + mean
    return tensor * std + mean




class Blend(nn.Module):
    def __init__(self, num_images, device='cpu'):
        super(Blend, self).__init__()
        # Initialize weights to be equal
        self.weights = nn.Parameter(torch.ones(num_images, device=device) / num_images)

    def forward(self, imgs):
        """
        imgs: [num_images, C, H, W]
        Returns:
            blended_img: [1, C, H, W]
        """
        # Normalize weights to sum to 1
        w_normalized = self.weights / self.weights.sum()
        # Reshape for broadcasting
        w_normalized = w_normalized.view(-1, 1, 1, 1)  # [num_images, 1, 1, 1]
        # Weighted sum
        blended_img = (imgs * w_normalized).sum(dim=0, keepdim=True)  # [1, C, H, W]
        return blended_img



def blend_DC(
    target_images,
    target_labels,
    model, 
    lr=0.05, 
    num_iterations=500, 
    device='cpu',
    batch_size=32
):
    

    """
    A blending-based distribution matching optimization.
    Each synthetic image is a weighted sum of target_images[i].
    """

    model.train()
    for param in model.parameters():
        param.requires_grad = False

    # ----------------------------------------------------
    # 1) Precompute target features for each target_image
    # ----------------------------------------------------
    target_features = []
    with torch.no_grad():
        for target_image in target_images:
            if batch_size is None:
                target_feature = model.embed(target_image.to(device)).detach()
                target_features.append(target_feature.cpu())

            else:
                target_feature = []
                for start_idx in range(0, len(target_image), batch_size):
                    end_idx = start_idx + batch_size
                    batch = target_image[start_idx:end_idx].to(device)
                    batch_features = model.embed(batch).detach()
                    target_feature.append(batch_features.cpu())

                target_feature = torch.cat(target_feature, dim=0)
                target_features.append(target_feature)

    # ----------------------------------------------------
    # 2) Create Blend modules for each target_images[i]
    #    Each Blend module has weights for target_images[i]
    # ----------------------------------------------------
    blend_modules = []
    num_target = len(target_images)
    for i in range(num_target):
        n_img = target_images[i].shape[0]
        blend_module = Blend(n_img, device=device).to(device)
        blend_modules.append(blend_module)

    # ----------------------------------------------------
    # 3) Set up optimizer for all Blend modules
    # ----------------------------------------------------


    optimizer = torch.optim.SGD(
        [param for blend in blend_modules for param in blend.parameters()],
        lr,
        momentum=0.5
    )


    # ----------------------------------------------------
    # 4) Main Optimization Loop
    # ----------------------------------------------------
    
    for it in range(num_iterations):

        dis_loss = torch.tensor(0.0, device=device)


        # Generate all synthetic images via blending
        for i, blend_module in enumerate(blend_modules):
            imgs_i = target_images[i].to(device)  # [n_i, C, H, W]
            synth_img = blend_module(imgs_i)     # [1, C, H, W]

            synth_feats_i = model.embed(synth_img)  # [num_target * K, feature_dim]

            dis_loss += dist_match(synth_feats_i, target_features[i].to(device))


        loss = dis_loss

        # Normalize by number of targets
        loss = loss / num_target

        print(f"Iteration {it+1}/{num_iterations}, Loss: {loss.item()}")

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ----------------------------------------------------
    # 5) Gather final synthetic images & labels
    # ----------------------------------------------------
    syn_images = []
    syn_labels = []
    with torch.no_grad():
        for i, blend_module in enumerate(blend_modules):
            final_img = blend_module(target_images[i].to(device))  # [1, C, H, W]
            # final_img = final_img - cifar_mean
            # final_img = final_img / cifar_std

            syn_images.append(final_img.cpu())

            # Grab the first label from target_labels[i]
            final_label = torch.tensor([target_labels[i][0].item()])
            syn_labels.append(final_label)

    syn_images_batch = torch.cat(syn_images, dim=0)  # [num_target, C, H, W]
    syn_labels_batch = torch.cat(syn_labels, dim=0)  # [num_target]



    return syn_images_batch, syn_labels_batch

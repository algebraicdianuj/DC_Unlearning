import warnings
warnings.filterwarnings('ignore')
import torch
from utils.dc_losses import *
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

class BatchedBlend(nn.Module):
    def __init__(self, batch_size, num_images_per_batch, bsizes, channels, height, width,device='cpu'):
        super(BatchedBlend, self).__init__()
        # Initialize weights to be equal
        self.weights = nn.Parameter(1/num_images_per_batch * torch.ones(batch_size, num_images_per_batch, device=device))
        self.channels = channels
        self.height = height
        self.width = width
        
        # Create a mask for trainable weights
        self.weight_mask = torch.ones_like(self.weights, device=device)
        
        # Zero out the weights for the padded images and make them untrainable
        for i, bsize in enumerate(bsizes):
            if bsize < num_images_per_batch:
                self.weights.data[i, bsize:] = 0
                self.weight_mask[i, bsize:] = 0


    def forward(self, imgs):
        # imgs shape: (batch_size, num_images_per_batch, channels, height, width)
        imgs = imgs.view(imgs.shape[0], imgs.shape[1], -1)
        
        # Apply the mask to ensure padded weights remain zero
        effective_weights = self.weights * self.weight_mask

        # normlaize weights
        effective_weights = effective_weights / effective_weights.sum(dim=1, keepdim=True)
        
        weighted_imgs = imgs * effective_weights.unsqueeze(-1)
        weighted_imgs = torch.sum(weighted_imgs, dim=1)
        weighted_imgs = weighted_imgs.reshape(imgs.shape[0], self.channels, self.height, self.width)
        return weighted_imgs

    def train(self, mode=True):
        super().train(mode)
        # Ensure the masked weights always have zero gradient
        if mode:
            self.weights.register_hook(lambda grad: grad * self.weight_mask)


def Average(ref_imgs_all_batched, bsizes, pretrained, channels = 3, height = 32, width = 32, lr = 1e-3,num_epochs=100, device = 'cpu'):

    ref_imgs_all_batched=ref_imgs_all_batched.to(device)
    weighted_avg_module = BatchedBlend(batch_size=ref_imgs_all_batched.shape[0],
                                                num_images_per_batch=ref_imgs_all_batched.shape[1],
                                                bsizes=bsizes,
                                                channels=channels,
                                                height=height, 
                                                width=width,
                                                device=device
                                                )
    
    # optim_weighted_avg = torch.optim.Adam(weighted_avg_module.parameters(), lr=lr)
    optim_weighted_avg = torch.optim.SGD(weighted_avg_module.parameters(), lr=lr, momentum=0.5)
    ref_features= pretrained.embed(ref_imgs_all_batched.view(-1, ref_imgs_all_batched.shape[2], ref_imgs_all_batched.shape[3], ref_imgs_all_batched.shape[4])).detach()

    for ep in range(num_epochs):
        fused_img_batch= weighted_avg_module(ref_imgs_all_batched)
        fused_img_features_batch= pretrained.embed(fused_img_batch)
        loss=torch.sum((torch.mean(ref_features, dim=0) - torch.mean(fused_img_features_batch, dim=0))**2)
        optim_weighted_avg.zero_grad()
        loss.backward()
        optim_weighted_avg.step()

    averaged_img=weighted_avg_module(ref_imgs_all_batched).detach()

    return averaged_img



def correct_sizes(collected_ref_imgs_all,chann, height, width):
    batch_sizes=[]
    for i in range(len(collected_ref_imgs_all)):
        batch_sizes.append(collected_ref_imgs_all[i].shape[0])

    max_batch_size=max(batch_sizes)
    for i in range(len(collected_ref_imgs_all)):
        if collected_ref_imgs_all[i].shape[0]<max_batch_size:
            diff=max_batch_size-collected_ref_imgs_all[i].shape[0]
            collected_ref_imgs_all[i]=torch.cat([collected_ref_imgs_all[i], torch.zeros(diff, chann, height, width)], dim=0)
            # collected_ref_imgs_all[i]=torch.cat([collected_ref_imgs_all[i], torch.ones(diff, chann, height, width)], dim=0)

    return collected_ref_imgs_all,batch_sizes


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

    channel, H, W = target_images[0].shape[1], target_images[0].shape[2], target_images[0].shape[3]
    
    
    collected_images = []
    blended_images = []
    blended_labels = []


    for i,target_image in enumerate(target_images):
        collected_images.append(target_image)
        blended_labels.append(target_labels[i][0])

        if len(collected_images) == batch_size:
            collected_images_all, batch_sizes = correct_sizes(collected_images, channel, H, W)
            real_images_all_batched = torch.stack(collected_images_all, dim=0)  # [batch_size, num_images_per_batch, C, H, W]

            blended_batch = Average(real_images_all_batched, batch_sizes, model, channels=channel, height=H, width=W, lr =lr, num_epochs=num_iterations, device=device)



            blended_images.append(blended_batch)
            collected_images = []  # Reset for the next batch

    Blended_images = torch.cat(blended_images, dim=0)  # [num_target, C, H, W]
    Blended_labels = torch.tensor(blended_labels)  # [num_target]



    return Blended_images, Blended_labels


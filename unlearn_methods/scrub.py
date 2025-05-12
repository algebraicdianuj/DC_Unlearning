import torch
import torch.nn as nn
import time
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR



class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = torch.nn.functional.log_softmax(y_s/self.T, dim=1)
        p_t = torch.nn.functional.softmax(y_t/self.T, dim=1)
        loss = torch.nn.functional.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def train_retain(epoch, retain_loader, model_s, model_t, criterion_list, optimizer, 
                gamma, beta, device, print_freq=12):
    """
    One epoch distillation on retain data to preserve knowledge
    """
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    
    
    model_s.train()
    model_t.eval()


    for idx, (input, target) in enumerate(retain_loader):
        input = input.to(device)
        target = target.to(device)
  
        # Forward pass
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # Calculate losses - minimize objective (retain knowledge)
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss = gamma * loss_cls + beta * loss_div
        

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_forget(epoch, forget_loader, model_s, model_t, criterion_list, optimizer, 
                device, print_freq=12):
    """
    One epoch distillation on forget data to encourage forgetting
    """
    criterion_div = criterion_list[1]
    
    
    model_s.train()
    model_t.eval()

 
    for idx, (input, target) in enumerate(forget_loader):
        input = input.to(device)
        target = target.to(device)

        # Forward pass
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # Calculate losses - maximize objective (forget knowledge)
        loss_div = criterion_div(logit_s, logit_t)
        loss = -0.2*loss_div  # Maximize divergence
        

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def scrub_model(
    teacher,
    student,
    retain_loader,
    forget_loader,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    warmup = 2,
    m_steps=1,
    epochs=10,
    kd_temp=4.0,
    gamma=0.1,
    beta=1.0,
    milestones=[5, 10, 15],
    device='cuda'
):
    """
    A single function that implements the scrubbing method for machine unlearning.
    
    Parameters:
        model (nn.Module): The model to be unlearned
        retain_loader (DataLoader): DataLoader for retain data
        forget_loader (DataLoader): DataLoader for forget data
        lr (float): Learning rate for optimization
        momentum (float): Momentum for SGD optimizer
        weight_decay (float): Weight decay for regularization
        m_steps (int): Number of epochs to perform maximize step
        epochs (int): Total number of epochs for unlearning
        kd_temp (float): Temperature for knowledge distillation
        gamma (float): Weight for classification loss
        beta (float): Weight for KL divergence loss
        milestones (list): Milestones for learning rate scheduler
        device (str): Device to run the model on
    
    Returns:
        model (nn.Module): The unlearned model
    """
    # Create a copy of the model for teacher
    teacher_model = teacher
    teacher_model.to(device)
    teacher_model.eval()
    
    # Set up the student model (the one that will be modified)
    student_model = student.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )


    # Use lambda scheduler with warmup and cosine annealing as in reference
    lambda0 = lambda cur_iter: (cur_iter + 1) / warmup if cur_iter < warmup else (
        0.5 * (1.0 + np.cos(np.pi * ((cur_iter - warmup) / (epochs - warmup))))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    # Set up criterion
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_temp)
    criterion_kd = DistillKL(kd_temp)
    

    
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)
    
 
    
    print(f"Retain loader length: {len(retain_loader)}, Forget loader length: {len(forget_loader)}")
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"Epoch #{epoch}, Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Forget step (make model forget)
        if epoch <= m_steps:
            train_forget(
                epoch, forget_loader, student_model, teacher_model, criterion_list, 
                optimizer, device
            )
        
        # Retain step (make model retain knowledge)
        train_retain(
            epoch, retain_loader, student_model, teacher_model, criterion_list, 
            optimizer, gamma, beta, device
        )
        
        # Update learning rate
        scheduler.step()
        

    return student_model


import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from utils.loading_model import get_model
from utils.utils import *
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
import random
import argparse
import os


import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
import time
import numpy as np
import random
import time
import pandas as pd
import torch
import pickle


from utils.hyperparameters import load_hyperparameters
from unlearn_methods.retrain import retraining
from unlearn_methods.distillation import distillation_unlearning
from unlearn_methods.sparisification import unlearn_with_l1_sparsity
from unlearn_methods.prunning import unlearn_with_pruning
from unlearn_methods.bad_distillation import blindspot_unlearner
from unlearn_methods.ntk_scrubbing import ntk_scrubbing
from unlearn_methods.fisher_forget import fisher_forgetting
from unlearn_methods.scrub import scrub_model
from unlearn_methods.ssd import ssd_unlearn
from unlearn_methods.ssd_lf import ssdlf_unlearn
from utils.loading_model import get_model
from utils.lira_mia import LiRA_MIA
import argparse



def main(args):
    print("-----------------------------------------------")
    print(f"Experiment: {args.exp}")
    print(f"Model Name: {args.model_name}")
    print(f"Unlearning Mode: Uniform")
    print("-----------------------------------------------")
    print("\n")

    directory_name= 'reservoir'
    current_path = os.getcwd()  
    data_storage = os.path.join(current_path, directory_name)  
    
    if not os.path.exists(data_storage): 
        os.makedirs(data_storage) 
        print(f"Directory '{directory_name}' created in the current working directory.")
    else:
        print(f"Directory '{directory_name}' already exists in the current working directory.")


    dat_dir='result'
    result_directory_path = os.path.join(current_path, dat_dir)

    if not os.path.exists(result_directory_path):
        os.makedirs(result_directory_path)
        print(f"Directory '{dat_dir}' created in the current working directory.")
    else:
        print(f"Directory '{dat_dir}' already exists in the current working directory.")

    #----------------------Hyperparameters---------------------------------
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


    batch_size = args.batch_size
    channel = 3
    im_size = torch.load(os.path.join(data_storage,'im_size.pt'))
    num_classes = torch.load(os.path.join(data_storage,'num_classes.pt'))



    do_retrain=True
    do_CF=True
    do_fisher=False
    do_ssd=False
    do_ssd_lf=False
    do_distillation=False
    do_scrub=True
    do_bad_distillation=True
    do_l1_sparsity=True
    do_pruning=True

    json_file_name=f'hyperparameters/{args.dataset}_{args.model_name}_hyperparameters.json'
    params = load_hyperparameters(json_file_name, args)
    globals().update(params)


    #------------------------------------------------------------------------



    #----------------------------Loading stuff------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()


    file_path = os.path.join(data_storage,'test_dataset.pth')
    dst_test = torch.load(file_path)

    file_path = os.path.join(data_storage,'train_dataset.pth')
    img_real_data_dataset = torch.load(file_path)

    img_real_data_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)


    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)



    #-----------------------------------Creating Subclasses---------------------------------------------------------
    train_images=img_real_data_dataset.images
    train_labels=img_real_data_dataset.labels

    num_samples = train_images.size(0)
    
    # Generate a random permutation of indices
    indices = torch.randperm(num_samples)

    # Shuffle training data using the shuffled indices
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    indices=list(range(len(train_images)))


    

    #---------------------Forget Set and Retain Set----------------------------------------------------------------------
    split_ratio = 0.1   # forget-retain split ratio (in case of random forgetting)

    retain_sets =[]
    forget_sets =[]

    all_retain_indices = []
    all_forget_indices = []
    choice = 'uniform'


    for i in range(args.unlearning_rounds):
        if i == 0:
            # Define split ratio and sizes
            split = int(split_ratio * len(img_real_data_dataset))

            # Initial split of indices
            forget_indices = indices[:split]
            retain_indices = indices[split:]

            # Create corresponding datasets
            forget_images = train_images[forget_indices]
            forget_labels = train_labels[forget_indices]
            retain_images = train_images[retain_indices]
            retain_labels = train_labels[retain_indices]

            print(f'Round {i} | Ratio of Forget to Retain: {len(forget_indices) / len(retain_indices):.4f}')
            print(f'Forget Set Size: {len(forget_indices)}')
            print(f'Retain Set Size: {len(retain_indices)}')
            print("------------------------------------------------")

            # Store datasets correctly
            forget_sets.append(TensorDatasett(forget_images, forget_labels))
            retain_sets.append(TensorDatasett(retain_images, retain_labels))

            # Track indices
            all_forget_indices.append(forget_indices)
            all_retain_indices.append(retain_indices)

        else:
            prior_retain_indices = all_retain_indices[-1]

            # Define new split
            split = int(split_ratio * len(prior_retain_indices))

            # Subdivide prior retain indices
            new_forget_indices = prior_retain_indices[:split]
            new_retain_indices = prior_retain_indices[split:]

            # Get corresponding data
            forget_images = train_images[new_forget_indices]
            forget_labels = train_labels[new_forget_indices]
            retain_images = train_images[new_retain_indices]
            retain_labels = train_labels[new_retain_indices]

            print(f'Round {i} | Ratio of Forget to Retain: {len(new_forget_indices) / len(new_retain_indices):.4f}')
            print(f'Forget Set Size: {len(new_forget_indices)}')
            print(f'Retain Set Size: {len(new_retain_indices)}')
            print("------------------------------------------------")

            # Store datasets in correct order
            forget_sets.append(TensorDatasett(forget_images, forget_labels))
            retain_sets.append(TensorDatasett(retain_images, retain_labels))

            # Track indices
            all_forget_indices.append(new_forget_indices)
            all_retain_indices.append(new_retain_indices)

 

    for round_num, (forget_set, retain_set) in enumerate(zip(forget_sets, retain_sets)):
        forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=batch_size, shuffle=True)
        retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=batch_size, shuffle=True)

        
        if do_retrain:
            #--------------------------Retraining Method------------------------------------------------------------------------------------
            naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
            starting_time = time.time()
            naive_net= retraining(naive_net, 
                                criterion, 
                                device, 
                                retrain_lr, 
                                retrain_momentum,
                                retrain_weight_decay,
                                retrain_warmup,
                                retrain_epochs, 
                                retain_loader,
                                decreasing_lr="50,75"
                                )
            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(naive_net, forget_loader,test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(naive_net, retain_loader, device)
            forget_acc=test(naive_net, forget_loader, device)
            test_acc=test(naive_net, test_loader, device)

            print('\nRetraining-Unlearning Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'retraining', df, args.model_name, args.exp, choice, round_num)

        

        if do_CF:

            #----------------------Catastrophic Forgetting Method---------------------------------------------------------------------------------
            naive_net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            starting_time = time.time()
            naive_net= retraining(naive_net, 
                                criterion, 
                                device, 
                                cf_lr, 
                                cf_momentum,
                                cf_weight_decay,
                                cf_warmup,
                                cf_epochs, 
                                retain_loader,
                                decreasing_lr="50,75"
                                )
            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(naive_net, forget_loader,test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(naive_net, retain_loader, device)
            forget_acc=test(naive_net, forget_loader, device)
            test_acc=test(naive_net, test_loader, device)


            print('\nCatastrophic-Forgetting Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'CF', df, args.model_name, args.exp, choice, round_num)





        if do_distillation:
            #-----------------------Distillation based Method---------------------------------------------------------------------------------
            teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            student_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            starting_time = time.time()
            distilled_net = distillation_unlearning(retain_loader, 
                                                    distill_lr ,
                                                    distill_momentum,
                                                    distill_weight_decay,
                                                    student_net, 
                                                    teacher_net, 
                                                    distill_epochs, 
                                                    device, 
                                                    alpha=distill_hard_weight, 
                                                    gamma=distill_soft_weight, 
                                                    kd_T=distill_kdT,
                                                    decreasing_lr="50,75"
                                                    )
            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(distilled_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(distilled_net, retain_loader, device)
            forget_acc=test(distilled_net, forget_loader, device)
            test_acc=test(distilled_net, test_loader, device)

            print('\nDistillation-Unlearning Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'distillation', df, args.model_name, args.exp, choice, round_num)



        if do_scrub:
            #-----------------------SCRUB Method---------------------------------------------------------------------------------
            student_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            starting_time = time.time()

            distilled_net = scrub_model(
                                        teacher_net,
                                        student_net,
                                        retain_loader,
                                        forget_loader,
                                        lr=scrub_lr,
                                        momentum=scrub_momentum,
                                        weight_decay=scrub_weight_decay,
                                        warmup= scrub_warmup,
                                        m_steps=scrub_msteps,
                                        epochs=scrub_epochs,
                                        kd_temp=scrub_kd_T,
                                        gamma=scrub_gamma,
                                        beta=scrub_beta,
                                        milestones=[5, 10, 15],
                                        device=device
                                    )
            

            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(distilled_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(distilled_net, retain_loader, device)
            forget_acc=test(distilled_net, forget_loader, device)
            test_acc=test(distilled_net, test_loader, device)

            print('\nSCRUB-Unlearning Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'SCRUB', df, args.model_name, args.exp, choice, round_num)


        if do_bad_distillation:
            #-----------------------Bad Teacher based Distillation Method---------------------------------------------------------------------------------
            teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            student_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            bad_teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
            starting_time = time.time()
            # sample the retain_set_real by partial_retain_ratio (for bad distillation)
            partial_retain_set_real = torch.utils.data.Subset(retain_set, random.sample(range(len(retain_set)), int(len(retain_set)*partial_retain_ratio)))
            distilled_net=blindspot_unlearner(model=student_net, 
                                            unlearning_teacher=bad_teacher_net, 
                                            full_trained_teacher=teacher_net, 
                                            retain_data=partial_retain_set_real,
                                            forget_data=forget_set, 
                                            epochs = bad_distill_epochs,
                                            lr = bad_distill_lr,
                                            momentum = bad_momentum,
                                            weight_decay = bad_distill_weight_decay,
                                            batch_size = batch_size, 
                                            device = device, 
                                            KL_temperature = bad_kdT
                                            )
            
            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(distilled_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(distilled_net, retain_loader, device)
            forget_acc=test(distilled_net, forget_loader, device)
            test_acc=test(distilled_net, test_loader, device)

            print('\nBad Teacher Distillation-Unlearning Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'Bad_distillation', df, args.model_name, args.exp, choice, round_num)


        if do_l1_sparsity:
            #-----------------------Sparsification based Method---------------------------------------------------------------------------------
            naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            starting_time = time.time()
            sparsified_net = unlearn_with_l1_sparsity(
                                                naive_net, 
                                                retain_loader,
                                                test_loader=None,
                                                unlearn_epochs=l1_epochs, 
                                                learning_rate=l1_lr,
                                                momentum=l1_momentum, 
                                                weight_decay=l1_weight_decay,
                                                alpha=l1_alpha,
                                                no_l1_epochs=l1_no_l1_epochs,
                                                warmup=l1_warmup,
                                                decreasing_lr="50,75",
                                                print_freq=50,
                                                device=device)
            

            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(sparsified_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(sparsified_net, retain_loader, device)
            forget_acc=test(sparsified_net, forget_loader, device)
            test_acc=test(sparsified_net, test_loader, device)

            print('\nSparsification-Unlearning Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'L1_sparsity', df, args.model_name, args.exp, choice, round_num)



        if do_pruning:
            #-----------------------Pruning based Method---------------------------------------------------
            naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
            starting_time = time.time()

            pruned_net = unlearn_with_pruning(
                                                naive_net,
                                                retain_loader, 
                                                unlearn_epochs=prune_epochs, 
                                                target_sparsity=prune_target_sparsity, 
                                                learning_rate=prune_lr,
                                                momentum = prune_momentum, 
                                                weight_decay=prune_weight_decay,
                                                prune_step=prune_step,
                                                decreasing_lr="50,75",
                                                print_freq=50,
                                                device=device)

            ending_time = time.time()
            unlearning_time=ending_time - starting_time

            mia_score=LiRA_MIA(pruned_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
            retain_acc=test(pruned_net, retain_loader, device)
            forget_acc=test(pruned_net, forget_loader, device)
            test_acc=test(pruned_net, test_loader, device)

            print('\nPrunning-Unlearning Stats: ')
            print('======================================')
            print('mia_score: ', mia_score)
            print('retain_acc: ', retain_acc)
            print('forget_acc: ', forget_acc)
            print('test_acc: ', test_acc)
            print('unlearning_time: ', unlearning_time)
            print('======================================')

            stat_data = {
                'mia_score': [mia_score],
                'retain_acc': [retain_acc],
                'forget_acc': [forget_acc],
                'test_acc': [test_acc],
                'unlearning_time': [unlearning_time]
            }

            df = pd.DataFrame(stat_data)

            save_data3(result_directory_path, 'pruning', df, args.model_name, args.exp, choice, round_num)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=1, help="Experiment number (default: 1)")
    parser.add_argument('--model_name', type=str, choices=['cnn_s', 'vit_s', 'resnet_s', 'resnetlarge_s'], required=True, 
                        help="Choose the model name from: vit, resnet, resnetlarge, vit_s, resnet_s, resnetlarge_s")
    parser.add_argument('--retrain_lr', type=float, default=1e-1, help="Retrain learning rate (default: 1e-4)")
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'svhn', 'cinic10'], default='cifar10', 
                        help="Choose the dataset from: cifar10 , cinic10, cifar100 or svhn")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: 128)")
    parser.add_argument('--unlearning_rounds', type=int, default=3, help="Number of Rounds of Unlearning (default: 3)")
    
    
    args = parser.parse_args()

    main(args)

# Leveraging Distribution Matching to Make Approximate Machine Unlearning Faster


<h2>Abstract</h2>

<div style="background-color:#2f2f2f; color:#f0f0f0; padding:16px; border-radius:8px; font-size:15px; line-height:1.5em;">
Approximate machine unlearning (AMU) enables models to <i>forget</i> specific training data through specialized fine-tuning on a retained dataset subset. However, processing this retained subset still dominates computational runtime, while reductions of epochs also remain a challenge. We propose two complementary methods to accelerate classification-oriented AMU. <b>Blend</b>, a novel distribution-matching dataset condensation (DC), merges visually similar images with shared blend-weights to significantly reduce the retained set size. It operates with minimal pre-processing overhead and is orders of magnitude faster than state-of-the-art DC methods. Our second method, <b>Accelerated-AMU (A-AMU)</b>, augments the unlearning objective to quicken convergence. A-AMU achieves this by combining a steepened primary loss to expedite forgetting with a novel, differentiable regularizer that matches the loss distributions of forgotten and in-distribution unseen data. Together, these methods dramatically reduce unlearning latency across both single and multi-round settings while preserving model utility and privacy.
</div>




## Preliminary
### Install Dependencies
```code
git clone https://github.com/algebraicdianuj/DC_U.git && cd DC_U
conda create -n torcher python=3.8.19
conda activate torcher
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -U scikit-learn
conda install scikit-image
conda install -c conda-forge opacus
pip install timm
```



### Download Datasets and Get Pretrained Models
#### CIFAR-10
```code
./scripts/get_cifar10.sh
./scripts/run_train.sh
```

#### SVHN
```code
./scripts/get_svhn.sh
./scripts/run_train.sh
```

#### CINIC-10
```code
./scripts/get_cinic10.sh
./scripts/run_train.sh
```




## Experiment: Section 4.2
```code
./scripts/run_c_cifar10.sh
```



## Experiment: Section 4.3

### CIFAR-10
#### Without Condensation
```code
./scripts/run_u_cifar10.sh
```
#### With Condensation
```code
./scripts/run_cu_cifar10.sh
```
### SVHN
#### Without Condensation
```code
./scripts/run_u_svhn.sh
```

#### With Condensation
```code
./scripts/run_cu_svhn.sh
```



## Experiment: Section 4.4

### CINIC-10
Below are the different ablation configurations tested:

| ablative_ | Description |
|---------|-------------|
| v1.py | Condensing whole retain |
| v2.py | Residual retain dataset|
| v3.py | Condensing free images |
| v4.py | Free images |
| v5.py | Condensing residual images |
| v6.py | Residual images |
| v7.py | Retain images |

```code
./scripts/ablations.sh
```


## Experiment: Section 4.5
### CINIC-10
#### Without Condensation
```code
./scripts/multiround_unlearn.sh
```

#### With Condensation
```code
./scripts/multiround_cond_unlearn.sh
```




## SOTA Unlearning Implementation References Used in this Repo

- [Fischer Forgetting](https://github.com/AdityaGolatkar/SelectiveForgetting)

- [NTK Scrubbing](https://github.com/AdityaGolatkar/SelectiveForgetting)

- [Prunning and Sparsity driven Catastrophic Forgetting](https://github.com/OPTML-Group/Unlearn-Sparse)

- [Distillation based Unlearning](https://github.com/meghdadk/SCRUB)

- [Good and Bad Teacher Distillation based Unlearning](https://github.com/vikram2000b/bad-teaching-unlearning)

- [Synaptic Forgetting](https://github.com/if-loops/selective-synaptic-dampening)



## SOTA Dataset Condensation References Used in this Repo
- [Dataset condensation with gradient matching](https://github.com/VICO-UoE/DatasetCondensation)
  
- [Dataset condensation with distribution matching](https://github.com/VICO-UoE/DatasetCondensation)
  
- [Improved distribution matching for dataset condensation](https://github.com/uitrbn/IDM)



## Further References Used in this Repo
- [Dataset Condensation Driven Machine Unlearning](https://github.com/algebraicdianuj/DC_U)


## Hyperparameters Report
### CIFAR10 | ResNet-18 | Dataset Condensation Performed: Yes

---


| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.04, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound: 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr 5e-3, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 9, scrub\_lr: 5e-3, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 20, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 5e-3, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-4, 11\_epochs: 30, 11\_no\_11\_epochs: 1, 11\_lr: 5e-3, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.04, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### CIFAR10 | ResNet-18 | Dataset Condensation Performed: No

| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.04, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound: 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr: 0.04, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 5, scrub\_lr: 1e-2, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 20, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 1e-2, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-4, 11\_epochs: 30, 11\_no\_l1\_epochs: 1, l1\_lr: 1e-2, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.04, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### CIFAR10 | ResNet-50 | Dataset Condensation Performed: Yes

| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.03, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr 5e-3, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 9, scrub\_lr: 5e-3, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 30, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 0.003, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-5, 11\_epochs: 40, 11\_no\_11\_epochs: 3, l1\_lr: 5e-3, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.03, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### CIFAR10 | ResNet-50 | Dataset Condensation Performed: No


| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.03, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr: 0.03, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 5, scrub\_lr: 1e-2, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 30, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 0.005, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-5, 11\_epochs: 40, 11\_no\_11\_epochs: 3, l1\_lr: 1e-2, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.03, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### CINIC10 | ResNet-18 | Dataset Condensation Performed: Yes

---


| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 1e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| asparse\_unlearning | af\_epochs: 1, asparse\_lr: 1e-4, asparse\_weight\_distribution: 34.0, asparse\_l1\_alpha: 1e-4, asparse\_k: 2.0, asparse\_K: 60.0 |
| ascrub\_unlearning | af\_epochs: 1, ascrub\_lr: 1e-4, ascrub\_weight\_distribution: 34.0, ascrub\_beta: 0.001, ascrub\_gamma: 0.99, ascrub\_kd\_T: 2.0, ascrub\_k: 2.0, ascrub\_K: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.04, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound: 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf : 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr : 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdt: 2.0, distill\_lr: 5e-3, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 10, scrub\_lr: 1e-3, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 20, partial\_retain\_ratio: 0.7234455830656301, bad\_distill\_lr: 5e-3, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-4, 11\_epochs: 30, 11\_no\_11\_epochs: 1, 11\_lr: 5e-3, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.04, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### CINIC10 | ResNet-18 | Dataset Condensation Performed: No

---



| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 1e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| asparse\_unlearning | af\_epochs: 1, asparse\_lr: 1e-4, asparse\_weight\_distribution: 34.0, asparse\_l1\_alpha: 1e-4, asparse\_k: 2.0, asparse\_K: 60.0 |
| ascrub\_unlearning | af\_epochs: 1, ascrub\_lr: 1e-4, ascrub\_weight\_distribution: 34.0, ascrub\_beta: 0.001, ascrub\_gamma: 0.99, ascrub\_kd\_T: 2.0, ascrub\_k: 2.0, ascrub\_K: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.04, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound: 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf : 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr : 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdt: 2.0, distill\_lr: 0.04, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 5, scrub\_lr: 5e-3, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 20, partial\_retain\_ratio: 0.7234455830656301, bad\_distill\_lr: 5e-3, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-4, 11\_epochs: 30, 11\_no\_11\_epochs: 1, 11\_lr: 1e-2, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.01, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### SVHN | ResNet-18 | Dataset Condensation Performed: Yes

---

| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.3, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound: 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr 5e-3, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 9, scrub\_lr: 5e-3, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 20, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 5e-3, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 8e-4, 11\_epochs: 30, 11\_no\_11\_epochs: 1, 11\_lr: 5e-3, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.04, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### SVHN | ResNet-18 | Dataset Condensation Performed: No


| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.04, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr: 0.04, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 5, scrub\_lr: 1e-2, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 20, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 1e-2, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-4, 11\_epochs: 30, 11\_no\_l1\_epochs: 1, l1\_lr: 1e-2, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.04, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### SVHN | ResNet-50 | Dataset Condensation Performed: Yes


| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.12, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 3 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr 5e-3, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 9, scrub\_lr: 5e-3, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 30, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 0.004, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 5e-4, 11\_epochs: 40, 11\_no\_11\_epochs: 3, l1\_lr: 5e-3, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.03, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |

---

### SVHN | ResNet-50 | Dataset Condensation Performed: No



| Method | Hyperparameters |
| :--- | :--- |
| acf\_unlearning | af\_epochs: 1, acf\_lr: 2.4e-4, weight\_distribution: 34.0, k: 2.0, κ: 60.0 |
| retraining | retrain\_lr: null, retrain\_epochs: 100, retrain\_momentum: 0.9, retrain\_weight\_decay: 0.0005, retrain\_warmup: 5 |
| catastrophic\_forgetting | cf\_lr: 0.03, cf\_epochs: 50, cf\_momentum: 0.502, cf\_weight\_decay: 0.0014, cf\_warmup: 5 |
| fisher\_forgetting | fisher\_alpha: 1e-6 |
| ssd | selective\_weighting: 10.0, dampening\_constant: 1.0, exponent: 1.0, lower\_bound 1.0, ssd\_lr: 0.04 |
| ssd\_If | selective\_weighting\_lf: 10.0, dampening\_constant\_lf: 1.0, exponent\_lf: 1.0, lower\_bound\_lf: 1.0, ssd\_lf\_lr: 0.04 |
| distillation | distill\_epochs: 50, distill\_hard\_weight: 1.0, distill\_soft\_weight: 0.001, distill\_kdT: 2.0, distill\_lr: 0.03, distill\_momentum: 0.86, distill\_weight\_decay: 0.001 |
| scrub | scrub\_beta: 0.001, scrub\_gamma: 0.99, scrub\_kd\_T: 2.0, scrub\_epochs: 15, scrub\_msteps: 5, scrub\_lr: 1e-2, scrub\_momentum: 0.9, scrub\_weight\_decay: 0.0005, scrub\_warmup: 5 |
| bad\_distillation | bad\_kdT: 2.0, bad\_distill\_epochs: 30, partial\_retain\_ratio: 0.6234455830656301, bad\_distill\_lr: 0.005, bad\_momentum: 0.9, bad\_distill\_weight\_decay: 0.0005 |
| 11\_sparsity | 11\_alpha: 1e-5, 11\_epochs: 40, 11\_no\_11\_epochs: 3, l1\_lr: 1e-2, 11\_momentum: 0.9, 11\_weight\_decay: 0.0005, 11\_warmup: 0 |
| pruning | prune\_lr: 0.03, prune\_epochs: 20, prune\_target\_sparsity: 0.95, prune\_weight\_decay: 0.0005, prune\_momentum: 0.9, prune\_step: 5 |






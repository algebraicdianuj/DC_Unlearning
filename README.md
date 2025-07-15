# Leveraging Distribution Matching to Make Approximate Machine Unlearning Faster
[![arXiv](https://img.shields.io/badge/arXiv-2507.09786-b31b1b.svg)](https://arxiv.org/abs/2507.09786)

![Distribution Matching for Machine Unlearning](./main_proposal.png)

<h2>Abstract</h2>

<div style="background-color:#2f2f2f; color:#f0f0f0; padding:16px; border-radius:8px; font-size:15px; line-height:1.5em;">
Approximate machine unlearning (AMU) enables models to <i>forget</i> specific training data through specialized fine-tuning on a retained dataset subset. However, processing this retained subset still dominates computational runtime, while reductions of epochs also remain a challenge. We propose two complementary methods to accelerate classification-oriented AMU. <b>Blend</b>, a novel distribution-matching dataset condensation (DC), merges visually similar images with shared blend-weights to significantly reduce the retained set size. It operates with minimal pre-processing overhead and is orders of magnitude faster than state-of-the-art DC methods. Our second method, <b>Accelerated-AMU (A-AMU)</b>, augments the unlearning objective to quicken convergence. A-AMU achieves this by combining a steepened primary loss to expedite forgetting with a novel, differentiable regularizer that matches the loss distributions of forgotten and in-distribution unseen data. Together, these methods dramatically reduce unlearning latency across both single and multi-round settings while preserving model utility and privacy.
</div>




## Preliminary
### Install Dependencies
```code
git clone https://github.com/algebraicdianuj/DC_U.git && cd DC_U
conda create -n DCU python=3.8.19
conda activate DCU
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



## SOTA Dataset Condensation References Used in this Repo
- [Dataset condensation with gradient matching](https://github.com/VICO-UoE/DatasetCondensation)
  
- [Dataset condensation with distribution matching](https://github.com/VICO-UoE/DatasetCondensation)
  
- [Improved distribution matching for dataset condensation](https://github.com/uitrbn/IDM)





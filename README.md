# INSA Project - HDDL

## Repository structure

```text
main/
├── datasets/
├   └── dataset_project_1/
├        ├──  images/
├        └── annotations/
├            ├──  list.txt
├            ├──  test.txt
├            ├──  trainval.txt
├            ├──  trimaps/
├            └──  xmls/
├   └── dataset_project_3/
├        ├──  cifar-10/
├            ├──  cifar-32/
├            └──  cifar-224/
├        └── food-101/
├── models/
├   ├── models_project_1.py
├   ├── models_project_3.py
├── utils/
├── weights/
├── demo_project_1.ipynb
├── demo_project_2.ipynb
├── demo_project_3.ipynb
└── requirements.txt
```


## Project 1 : Classification & Segmentation : Cats vs Dogs
[Acess to demo_project_1](https://github.com/selim-ba/insa-hddl-group10/blob/main/demo_project_1.ipynb)

**Our notebook is structured as follows :**

1. **Exploratory Data Analysis**

2. **Binary (Species) Classification**
- CNN vs Pre-Trained + Fine-Tuned ResNet-18

| Task | Criterion | Optimizer | LR Scheduler | Model | Augmentation | Epochs | Train Loss | Val Loss | Train Acc | Val Acc | ROC AUC | PR AUC (AP) | Precision Cat | Recall Cat | Precision Dog | Recall Dog | F1-Score Cat | F1-Score Dog |Macro Precision | Macro Recall | Macro F1 |
|------|-----------|-----------|--------|-----------|-----------|--------------|--------|------------|----------|-----------|----------|----------|--------------|----------------|-------------|----------------|-------------|---------------|----------------| ----------------| ----------------|
| Binary Clf | CrossEntropyLoss | Adam (lr=1e-4, wd=1e-4) | No | CNN_Binary | None | 30 | 0.1661 | 0.3793 | 0.933 | 0.848 | 0.924 | 0.963 | 0.709 | 0.843 | 0.926 | 0.850 | 0.770 | 0.886 | 0.817 | 0.846 | 0.828 |
| Binary Clf | CrossEntropyLoss | Adam (lr=3e-5, wd=1e-4) | No | CNN_Binary | None | 30 | 0.3295 | 0.5270 | 0.859 | 0.793 | 0.866 | 0.914 | 0.883 | 0.414 | 0.563 | 0.777 | 0.974 | 0.865 | 0.830 | 0.694 | 0.714 |
| Binary Clf | CrossEntropyLoss | Adam (lr=1e-4, wd=1e-4) | step_size=10, gamma = 0.1 | CNN_Binary | None | 30 | 0.3440 | 0.4041 | 0.856 | 0.822 | 0.871 | 0.939 | 0.728 | 0.659 | 0.857 | 0.893 | 0.692 | 0.875 | NA | NA | NA |
| Binary Clf | CrossEntropyLoss (with balanced weights) | Adam (lr=1e-4, wd=1e-4) | No | CNN_Binary | None | 30 | 0.2015 | 1.5069 | 0.922 | 0.755 | 0.899 | 0.952 | 0.939 | 0.206 | 0.742 | 0.994 | 0.338 | 0.850 |  0.840 | 0.600 | 0.594 |
| Binary Clf | CrossEntropyLoss | Adam (lr=1e-4, wd=1e-4) | No | CNN_Binary | Yes | 30 | 0.2865 | 0.3123 | 0.880 | 0.860 | 0.930 | 0.969 | 0.853 | 0.650 | 0.862 | 0.951 | 0.738 | 0.904 |  0.857 | 0.801 | 0.821 |
| ResNet18 | CrossEntropyLoss | AdamW (Stage 1: head lr=1e-4; Stage 2: backbone=1e-5, head=1e-4) | No | ResNet18_Binary | No | 5 + 30 | 0.0007 | 0.0314 | 1.0 | 0.992 | 0.999 | 0.999 | 0.975 | 1.0 | 1.0 | 0.988 | 0.988  | 0.994 |  0.988 | 0.994 | 0.991 |
| ResNet18 | CrossEntropyLoss | AdamW (Stage 1: head lr=1e-4; Stage 2: backbone=1e-5, head=1e-4) | No | ResNet18_Binary | Yes | 5 + 30 | 0.0054 | 0.0075 | 0.998 | 0.996 | 1.0 | 1.0 | 1.0 | 0.987 | 0.994 | 1.0 | 0.994 | 0.997 | 0.997  | 0.994 |  0.995 |

3. **Multiclass (Breeds) Classification**
- CNN vs Pre-Trained + Fine-Tuned VGG-16

|Task | Criterion | Optimizer | Model | Augmentation | Epochs | Train Loss | Val Loss | Train Acc | Val Acc | ROC AUC | Weighted Precision | Weighted Recall | Weighted F1 | Most confused breeds|
|------|-----------|--------|--------------|--------------|--------|----------|-----------|----------|----------|--------------|--------------|----------------|-------------|------------|
| Multiclass classification | CrossEntropyLoss | AdamW(lr=1e-4, weight_decay=1e-4) | CNN_multiclass | No | 30 | 0.0001 | 8.7 | 1.0 | 0.19 | 0.75 | 0.183 | 0.189 | 0.183 | Bengal vs. Yorkshire Terrier (0.3) | 
| Multiclass classification | CrossEntropyLoss | AdamW(lr=1e-4, weight_decay=1e-4) | CNN_multiclass | Yes | 30 | 0.5155 | 4.2479 | 0.841 | 0.287 | 0.850 | 0.305 | 0.287 | 0.281 | Samoyed vs. Beagle (0.4) | 
| Multiclass classification | CrossEntropyLoss |  AdamW (Stage 1: head lr=1e-4; Stage 2: backbone=1e-5, head=1e-4)| VGG-16 | No | 5 + 30 | 0.0023 | 0.395 | 0.999| 0.931 | 0.99 | 0.934 | 0.931 | 0.929 | Ragadoll vs. Birman (0.25) | 
| Multiclass classification | CrossEntropyLoss |  AdamW (Stage 1: head lr=1e-4; Stage 2: backbone=1e-5, head=1e-4)| VGG-16 | Yes | 5 + 30 | 0.0160 | 0.3413| 0.993 | 0.931 | 0.99 | 0.934| 0.931 | 0.930 | Ragadoll vs. Birman (0.25) | 

4. **Semantic Segmentation & Comparative Analysis**
- U-Net vs U-Net++

| Task | Criterion | Optimizer | Model | Augmentation | Epochs | Train Loss | Val Loss | Train Acc | Val Acc | Train Dice | Val Dice | Train IoU | Val IoU | 
|------|-----------|-----------|--------|-----------|--------------|--------|------------|----------|-----------|----------|----------|--------------|----------------|
| Segmentation | CrossEntropyLoss | Adam (lr=1e-3, wd=1e-4) |  U-Net | Yes | 20 | 0.1985 | 0.2331 | 0.911 | 0.915 | 0.863 | 0.855 | 0.774 | 0.763 |
| Segmentation | Dice Loss | Adam (lr=1e-4, wd=1e-4) |  U-Net | Yes | 20 | 0.1441 | 0.1619 | 0.901 | 0.895 | 0.856 | 0.838 | 0.762 | 0.736 |
| Segmentation | CE (80%) + Dice Loss (20%) | Adam (lr=1e-3, wd=1e-4) |  U-Net | Yes | 20 | 0.2512 | 0.2810 | 0.890 | 0.889 | 0.835 | 0.826 | 0.734 | 0.720 | 
| Segmentation | CE | Adam (lr=1e-3, wd=1e-4) |  UNet++ | Yes | 20 | 0.2575 | 0.2735 | 0.889 | 0.896 | 0.832 | 0.827 | 0.730 | 0.725 |
| Segmentation | CE | Adam (lr=1e-3, wd=1e-4) |  UNet++ | Yes | 50 | 0.2153 | 0.2788 | 0.905 | 0.899 | 0.853 | 0.825 | 0.760 | 0.724 | 

---

| Task | Criterion |  Model | Trained on X epochs | Test Loss | Test Accuracy | Test Dice | Test IoU |
|------|-----------|-----------|--------|--------|--------------|--------| --------|
| Segmentation | CrossEntropyLoss | U-Net | 20 |  0.233 | 0.913 | 0.855 | 0.763 | 
| Segmentation | Dice Los | U-Net | 20 | 0.163 | 0.889 | 0.836 | 0.733 |
| Segmentation | CE (80%) + Dice Loss (20%) | U-Net | 20 | 0.286 | 0.886 | 0.825 | 0.719 | 
| Segmentation | CE | UNet++ | 20 | 0.288 |  0.890 | 0.823 | 0.719 |
| Segmentation | CE | UNet++ | 50 | 0.287 | 0.895 | 0.822 | 0.721 |

5. **Conclusion**

## Project 2 : Conditional VAE



## Project 3 : Classification : CNN vs ViT

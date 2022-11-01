# FeatureRE
This repository is the source code for ["Rethinking the Reverse-engineering of Trojan Triggers"](https://arxiv.org/abs/2210.15127) (NeurIPS 2022).

<div align="center">
<img src=./image/overview.png width=75% />
</div>

Existing reverse-engineering methods only consider the input space constraint. It conducts
reverse-engineering via searching a static trigger pattern in the input space. These methods fail to
reverse-engineer feature-space Trojans whose trigger is dynamic in the input space. Instead, our idea
is to exploit the feature space constraint and searching a feature space trigger using the constraint
that the Trojan features will form a hyperplane. At the same time, we also reverse-engineer the input
space Trojan transformation based on the feature space constraint. 

## Environment
See requirements.txt

## Generating models
Trojaned models can be generated via using the existing code of the attacks:

- [BadNets] https://github.com/verazuo/badnets-pytorch  
- [WaNet] https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release  
- [IA] https://github.com/VinAIResearch/input-aware-backdoor-attack-release  
- [CL] https://github.com/MadryLab/label-consistent-backdoor-code  
- [Filter] https://github.com/trojai  
- [SIG] https://github.com/bboylyg/NAD  
- [ISSBA] https://github.com/yuezunli/ISSBA  

For example, to generate Trojaned models by WaNet:
```bash
cd train_models \
CUDA_VISIBLE_DEVICES=0 python train_model.py --dataset cifar10 --set_arch resnet18 --pc 0.1
```
To generate benign models:
```bash
cd train_models \
CUDA_VISIBLE_DEVICES=0 python train_model.py --dataset cifar10 --set_arch resnet18 --pc 0
```

## Detection

For example, to run FeatureRE detection on CIFAR10 with ResNet18 network:

```bash
CUDA_VISIBLE_DEVICES=0 python detection.py \
--dataset cifar10 --set_arch resnet18 \
--hand_set_model_path <path_to_.pth_file> \
--data_fraction 0.01 \
--lr 1e-3 --bs 256 \
--set_all2one_target all
```

## Mitigation

For example, to run FeatureRE mitigation on CIFAR10 with ResNet18 network produced by filter attack:

```bash
CUDA_VISIBLE_DEVICES=0 python mitigation.py \
--dataset cifar10 --set_arch resnet18 \
--hand_set_model_path <path_to_.pth_file> \
--data_fraction 0.01 \
--lr 1e-3 --bs 256 \
--set_all2one_target <detected_target_label> \
--mask_size 0.05 --override_epoch 400 --asr_test_type wanet
```

## Cite this work
You are encouraged to cite the following paper if you use the repo for academic research.

```
@inproceedings{wang2022rethinking,
  title={Rethinking the Reverse-engineering of Trojan Triggers},
  author={Wang, Zhenting and Mei, Kai and Ding, Hailun and Zhai, Juan and Ma, Shiqing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

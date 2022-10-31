# FeatureRE
This repository is the source code for "Rethinking the Reverse-engineering of Trojan Triggers".

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
```
cd train_models \
CUDA_VISIBLE_DEVICES=0 python train_model.py --dataset cifar10 --set_arch resnet18 --pc 0.1
```
To generate benign models:
```
cd train_models \
CUDA_VISIBLE_DEVICES=0 python train_model.py --dataset cifar10 --set_arch resnet18 --pc 0
```

## Detection

For example, to run FeatureRE detection on CIFAR10 with ResNet18 network:

```
CUDA_VISIBLE_DEVICES=0 python detection.py \
--dataset cifar10 --set_arch resnet18 \
--hand_set_model_path <path_to_.pth_file> \
--data_fraction 0.01 \
--lr 1e-3 --bs 256 \
--set_all2one_target all
```

## Mitigation

For example, to run FeatureRE mitigation on CIFAR10 with ResNet18 network produced by filter attack:

```
CUDA_VISIBLE_DEVICES=0 python mitigation.py \
--dataset cifar10 --set_arch resnet18 \
--hand_set_model_path <path_to_.pth_file> \
--data_fraction 0.01 \
--lr 1e-3 --bs 256 \
--set_all2one_target <detected_target_label> \
--mask_size 0.05 --override_epoch 400 --asr_test_type wanet
```

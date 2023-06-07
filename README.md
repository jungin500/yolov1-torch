# yolov1-torch
YOLOv1 PyTorch from scratch

**See newer `v2` implementation on: [v2 branch](https://github.com/jungin500/yolov1-torch/tree/v2)**

# 1. Pretrained model (custom darknet imagnet classifier)
## 1.1. Overview
| Type            | Description | Notes       |
|-----------------|-------------|-------------|
| PyTorch Version | 1.8.1       | with CUDA   |
| OS              | Ubuntu      |             |
| GPU             | GTX 1080 Ti | VRAM 11GB   |
| Batch size      | 64          | VRAM 5.15GB |
| Optimizer       | Adam        |             |
- 기간: 2021/06/14(~current)

## 1.2. Hyperparameters & Environment
| Type              | Description                              | Notes                                                                           |
|-------------------|------------------------------------------|---------------------------------------------------------------------------------|
| Model             | custom pretrained model (darknet)        | pretrainer.py trains initial ~20 conv layers with ILSVRC2012 dataset            |
| Model Type        | Classifier                               | ILSVRC2012                                                                      |
| Autograd          | AMP autograd, GradScaler                 | Higher LR will cause gradient explosion                                         |
| Batch size        | 64                                       | Affects learning rate                                                           |
| BatchNorm         | False                                    | Not specified on paper                                                          |
| Input size        | 3*224*224                                | CHW for pytorch                                                                 |
| Input constraints | FP32, range normalized, image normalized | mean=[0.4547857, 0.4349471, 0.40525291] std=[0.12003352, 0.12323549, 0.1392444] |
| Output size       | 1000                                     | ILSVRC2012                                                                      |
| Optimizer         | Adam                                     |                                                                                 |
| Optimal LR-Decay  | 0.95                                     | every 20 epochs, also described in pretainer.py                                 |
| Optimal init LR   | 0.00025                                  | NOT specified on paper!                                                         |

## 1.3. Tensorboard result (~24 epcoh only)
![image](https://user-images.githubusercontent.com/5201073/122158320-71db7480-cea7-11eb-8e39-baf4ced7e206.png)

## 2. Detection model (full model, TBD)
## 2.1. Overview
| Type            | Description | Notes       |
|-----------------|-------------|-------------|
| PyTorch Version | 1.8.1       | with CUDA   |
| OS              | Windows 10  |             |
| GPU             | RTX 3070    | VRAM 8GB    |
| Batch size      | 1           | VRAM ~7.1GB |
| Optimizer       | Adam        |             |

## 2.2. Hyperparameters & Environment
| Type              | Description                              | Notes                                                                           |
|-------------------|------------------------------------------|---------------------------------------------------------------------------------|
| Model             | custom pretrained model (darknet)        | pretrainer.py trains initial ~20 conv layers with ILSVRC2012 dataset            |
| Model Type        | Classifier                               | ILSVRC2012                                                                      |
| Autograd          | AMP autograd, GradScaler                 | Higher LR will cause gradient explosion                                         |
| Batch size        | 64                                       | Affects learning rate                                                           |
| BatchNorm         | False                                    | Not specified on paper                                                          |
| Input size        | 3*224*224                                | CHW for pytorch                                                                 |
| Input constraints | FP32, range normalized, image normalized | mean=[0.4547857, 0.4349471, 0.40525291] std=[0.12003352, 0.12323549, 0.1392444] |
| Output size       | 1000                                     | ILSVRC2012                                                                      |
| Optimizer         | Adam                                     |                                                                                 |
| Optimal LR-Decay  | 0.95                                     | every 20 epochs, also described in pretainer.py                                 |
| Optimal init LR   | 0.00025                                  | NOT specified on paper!                                                         |

## 2.3. Inspections
- Windows 환경의 문제인지 모델의 문제인지 Batch size가 1인데도 8GB를 꽉 채워버림
- Batch size 1에 Learning rate 0.01에서 잘 작동함 (이후 higher batch size 환경에서 테스트 예정)

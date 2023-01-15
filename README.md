# You Only Look Once: Unified, Real-Time Object Detection
Implementation of paper "_You Only Look Once: Unified, Real-Time Object Detection, Redmon et al._" using [PyTorch Lightning](https://www.pytorchlightning.ai/), [ImageNet dataset](https://www.image-net.org/) and [VOC2007/2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

## Abstract
> We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.  
> Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is far less likely to predict false detections where nothing exists. Finally, YOLO learns very general representations of objects. It outperforms all other detection methods, including DPM and R-CNN, by a wide margin when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.

# Quick Start
TBD

# Usage
TBD

# Limitations
TBD

# Timeline
- [X] Implement YOLO model
- [X] Separate backbone,head from model
- [X] [Backbone] Implement ImageNet training pipeline
- [X] [Backbone] Fine-tuning hyperparameters
- [X] [Backbone] Train/Evaluate network for ~~a week~~ few days
- [ ] [Head] Implement VOC2007/2012 training pipeline
- [ ] [Head] Implement loss function
- [ ] [Head] Fine-tuning hyperparameters
- [ ] [Head] Train/Evaluate network for few days

# Why from scratch?
Building model architecture from scratch is primarily for reviewing my knowledge, but it could also be used as a baseline for paper implementation.

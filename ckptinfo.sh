#!/bin/bash
# Inspect checkpoint file
CKPT=$@
python3 -c "print('Loading libraries...'); import torch; print('Loading checkpoint file $CKPT ...'); ckpt=torch.load('$CKPT'); print('PyTorch Lightning checkpoint (Version {})\nEpoch: {} (Global step {})'.format(ckpt['pytorch-lightning_version'], ckpt['epoch'], ckpt['global_step']))"
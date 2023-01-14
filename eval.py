from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Resize, ToTensor, Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torchmetrics

from model import YOLOv1
from train import YOLOPretrainLitModel


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b',
                        '--batch-size',
                        default=32,
                        type=int,
                        help="Batch size")
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Toggle FP16 evaluation')
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        help='PyTorch lightning checkpoint location')
    parser.add_argument('dataset_root',
                        type=str,
                        help='ImageNet2012 dataset root path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    lit_model = YOLOPretrainLitModel.load_from_checkpoint(
        args.checkpoint,
        # Required parameters
        batch_size=args.batch_size,
        learning_rate=.0,
        num_workers=4,
        dataset_root=args.dataset_root,
    )

    lit_model.eval()
    lit_model.freeze()

    trainer = Trainer(
        precision=16 if args.fp16 else 32,
        accelerator='gpu',
        # Forcing 1 devices due to DDP sampler issue
        # See: https://pytorch-lightning.readthedocs.io/en/1.6.3/common/evaluation.html
        devices=1)

    trainer.validate(
        lit_model,
        verbose=True,
    )

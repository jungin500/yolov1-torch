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


class YOLOPretrainLitModel(LightningModule):

    def __init__(
        self,
        yolo_model: YOLOv1,
        batch_size: int,
        learning_rate: int,
        num_workers: int,
        dataset_root: str,
    ):
        super().__init__()
        self.yolo_model = yolo_model
        self.yolo_model.pretrain_mode = True

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.dataset_root = dataset_root

        self.train_acc = torchmetrics.Accuracy(num_classes=1000)
        self.val_acc = torchmetrics.Accuracy(num_classes=1000)

    def train_dataloader(self):
        dataset = ImageNet(self.dataset_root,
                           split='train',
                           transform=Compose([
                               RandomResizedCrop(224),
                               RandomHorizontalFlip(),
                               ToTensor(),
                               Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225],
                               )
                           ]))
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = ImageNet(self.dataset_root,
                           split='val',
                           transform=Compose([
                               Resize((224, 224)),
                               ToTensor(),
                               Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225],
                               )
                           ]))
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def configure_optimizers(self):
        return optim.Adam(self.yolo_model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.yolo_model(x)
        loss = nn.functional.cross_entropy(pred, y)
        acc = self.train_acc(pred, y)
        self.log('train/acc_step', acc)
        self.log('train/loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train/acc_epoch', self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.yolo_model(x)
        self.val_acc.update(pred, y)

    def validation_epoch_end(self, outputs):
        self.log('val/acc_epoch', self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b',
                        '--batch-size',
                        default=32,
                        type=int,
                        help="Batch size (divided if multi-gpu)")
    parser.add_argument('-lr',
                        '--learning-rate',
                        default=3e-5,
                        type=float,
                        help="Learning rate for optimizer")
    parser.add_argument('-j',
                        '--num-workers',
                        default=4,
                        type=int,
                        help='Num. of dataloader worker processes')
    parser.add_argument('-d',
                        '--devices',
                        default=1,
                        type=int,
                        help='Num. of GPU devices to train')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Toggle FP16 training')
    parser.add_argument('--wandb-token',
                        type=str,
                        default='',
                        help='Wandb token for login')
    parser.add_argument('--no-wandb',
                        action='store_true',
                        help='Disable wandb integration')
    parser.add_argument('dataset_root',
                        type=str,
                        help='ImageNet2012 dataset root path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.no_wandb:
        assert len(args.wandb_token
                   ) != 0, "Put wandb token in --wandb-token argument!"

        import wandb
        wandb.login(key=args.wandb_token)

    model = YOLOv1(pretrain_mode=True, )
    lit_model = YOLOPretrainLitModel(
        yolo_model=model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
    )

    trainer = Trainer(
        logger=True if args.no_wandb else WandbLogger(
            project="yolov1-imagenet"),
        precision=16 if args.fp16 else 32,
        max_epochs=200,
        accelerator='gpu',
        devices=args.devices,
        # ddp_spawn is default ddp strategy
        # but it is not recommended
        # See: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html
        strategy='ddp' if args.devices > 1 else None,
        callbacks=[
            # EarlyStopping('val/acc_epoch', patience=5),
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename='yolov1-backbone-{valid_acc_epoch:.2f}',
                monitor='val/acc_epoch',
                mode='max',
            )
        ])

    trainer.fit(lit_model)
    print("Training sequence done.")
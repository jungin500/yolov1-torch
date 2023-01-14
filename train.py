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
        batch_size: int,
        learning_rate: int,
        num_workers: int,
        dataset_root: str,
    ):
        super().__init__()
        self.yolo_model = YOLOv1(pretrain_mode=True, )
        self.yolo_model.pretrain_mode = True

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.dataset_root = dataset_root

        self.train_acc = torchmetrics.Accuracy(num_classes=1000)
        self.train_acc_top5 = torchmetrics.Accuracy(num_classes=1000, top_k=5)
        self.val_acc = torchmetrics.Accuracy(num_classes=1000)
        self.val_acc_top5 = torchmetrics.Accuracy(num_classes=1000, top_k=5)

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

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        return optim.Adam(self.yolo_model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.yolo_model(x)
        loss = nn.functional.cross_entropy(pred, y)
        acc = self.train_acc(pred, y)
        acc_top5 = self.train_acc_top5(pred, y)
        self.log('train/acc_step', acc)
        self.log('train/acc_top5_step', acc_top5)
        self.log('train/loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        train_acc_epoch = self.train_acc.compute()
        train_acc_top5_epoch = self.train_acc_top5.compute()
        self.train_acc.reset()
        self.train_acc_top5.reset()

        self.log(
            'train/acc_epoch',
            train_acc_epoch,
            sync_dist=True,
        )
        self.log(
            'train/acc_top5_epoch',
            train_acc_top5_epoch,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.yolo_model(x)
        self.val_acc.update(pred, y)
        self.val_acc_top5.update(pred, y)

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc.compute()
        val_acc_top5 = self.val_acc_top5.compute()
        self.val_acc.reset()
        self.val_acc_top5.reset()

        self.log(
            'val/acc',
            val_acc,
            sync_dist=True,
        )
        self.log(
            'val/acc_top5',
            val_acc_top5,
            prog_bar=True,
            sync_dist=True,
        )


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
    parser.add_argument('--no-wandb',
                        action='store_true',
                        help='Disable wandb integration')
    parser.add_argument('dataset_root',
                        type=str,
                        help='ImageNet2012 dataset root path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    lit_model = YOLOPretrainLitModel(
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
            # EarlyStopping('val/acc_top5', patience=5),
            ModelCheckpoint(
                dirpath='./checkpoints',
                auto_insert_metric_name=False,
                filename=
                'yolov1-backbone-epoch{epoch:04d}-step{step:06d}-val_acc_top5{val_acc_top5:.2f}',
                monitor='val/acc_top5',
                mode='max',
            )
        ])

    trainer.fit(lit_model)
    print("Training sequence done.")
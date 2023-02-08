from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Resize, ToTensor, Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize

from lightning import LightningModule
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
        optimizer = optim.Adam(
            params=self.yolo_model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            patience=5,
            threshold=1e-3,
            cooldown=5,
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "val/acc_top5",
        }

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

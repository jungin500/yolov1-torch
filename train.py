from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Resize, ToTensor, Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torchmetrics

from model import YOLOv1

batch_size = 256


class YOLOPretrainLitModel(LightningModule):

    def __init__(self, yolo_model: YOLOv1, dataset_root: str):
        super().__init__()
        self.yolo_model = yolo_model
        self.yolo_model.pretrain_mode = True
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
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8)

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
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8)

    def configure_optimizers(self):
        return optim.Adam(self.yolo_model.parameters(), lr=3e-5)

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


if __name__ == '__main__':
    model = YOLOv1(pretrain_mode=True, )
    lit_model = YOLOPretrainLitModel(
        yolo_model=model,
        dataset_root='dataset/imagenet',
    )

    trainer = Trainer(
        logger=WandbLogger(project="yolov1-imagenet"),
        precision=32,
        max_epochs=200,
        accelerator='gpu',
        devices=1,
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
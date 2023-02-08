import os
from xml.etree import ElementTree as ET
import cv2
from PIL import Image
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from torchvision.transforms import Resize, ToTensor, Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize

from lightning import LightningModule
from torchvision.ops import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model import YOLOv1


def yolo_det_collate(x):
    images = []
    labels = []
    for image, label in x:
        images.append(image)
        labels.append(label)

    images = torch.stack(images, 0)

    return images, labels


class VOCDataset(Dataset):

    def __init__(self,
                 dataset_root,
                 classes,
                 split='train',
                 transform=None,
                 target_transform=None):
        super(VOCDataset, self).__init__()
        self.dataset_root = dataset_root
        self._class_idmap = {
            class_name: idx
            for idx, class_name in enumerate(classes)
        }
        self.image_filenames, self.annotations = self._parse_annotation(
            dataset_root, split)
        self.transform = transform
        self.target_transform = target_transform

    def _class2id(self, class_name):
        return self._class_idmap[class_name]

    def _parse_single_annotation(self, xml_file):
        filename = None
        bboxes = []
        with open(xml_file) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            filename = root.find('filename').text
            img_width = int(root.find('size').find('width').text)
            img_height = int(root.find('size').find('height').text)
            for object_dom in root.findall('object'):
                class_id = self._class2id(object_dom.find('name').text)
                bbox_dom = object_dom.find('bndbox')
                xmin = int(float(bbox_dom.find('xmin').text))
                ymin = int(float(bbox_dom.find('ymin').text))
                xmax = int(float(bbox_dom.find('xmax').text))
                ymax = int(float(bbox_dom.find('ymax').text))

                xmin, xmax = xmin / img_width, xmax / img_width
                ymin, ymax = ymin / img_height, ymax / img_height
                width, height = xmax - xmin, ymax - ymin
                xcenter = xmin + (width / 2)
                ycenter = ymin + (height / 2)
                bboxes.append([xcenter, ycenter, width, height, class_id])

        return filename, bboxes

    def _parse_annotation(self, dataset_root, split):
        image_filenames = []
        annotations = []
        for filename_base in open(
                os.path.join(
                    dataset_root,
                    'ImageSets',
                    'Main',
                    '{}.txt'.format(split),
                ),
                encoding='utf-8',
        ).read().splitlines():
            xml_file = os.path.join(
                dataset_root,
                'Annotations',
                '{}.xml'.format(filename_base),
            )
            filename, annotation = self._parse_single_annotation(xml_file)
            filepath = os.path.join(dataset_root, 'JPEGImages', filename)
            image_filenames.append(filepath)
            annotations.append(annotation)

        return image_filenames, annotations

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], cv2.IMREAD_COLOR)
        print("Current image: {}".format(self.image_filenames[index]))
        
        #! DEBUG
        from pathlib import Path
        cv2.imwrite(Path(self.image_filenames[index]).name, image)
        #! END DEBUG

        if self.transform is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = cv2.resize(image, (416, 416))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = (image / 255.).astype(np.float32)
            # Normalize
            image -= np.array([0.485, 0.456, 0.406])[:, None, None]
            image *= np.array([0.229, 0.224, 0.225])[:, None, None]
            image = torch.from_numpy(image)

        label = self.annotations[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        else:
            label = torch.Tensor(label)
        return image, label


class YOLODetectorLitModel(LightningModule):

    def __init__(
        self,
        backbone_ckpt: str,
        batch_size: int,
        learning_rate: int,
        num_workers: int,
        dataset_root: str,
        test_root: str,
    ):
        super().__init__()
        self.yolo_model = YOLOv1(pretrain_mode=False, )
        self.yolo_model.pretrain_mode = False

        if backbone_ckpt is not None:
            backbone_dict = {
                self._replace_key(key): value
                for (key,
                     value) in torch.load(backbone_ckpt)['state_dict'].items()
            }
            self.yolo_model.load_state_dict(backbone_dict, strict=True)
            for param in self.yolo_model.backbone.parameters():
                param.requires_grad = False

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.dataset_root = dataset_root
        self.test_root = test_root

        self.train_mean_ap = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )
        self.val_mean_ap = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )

    def _replace_key(self, key):
        return key.replace('yolo_model.', '') \
            .replace('head.', 'detection_head.') \
            .replace('pretrain_detection_head.', 'pretrain_head.')

    def train_dataloader(self):
        dataset = VOCDataset(dataset_root=self.dataset_root,
                             classes=open('classes.txt').read().splitlines(),
                             split='trainval')
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=yolo_det_collate)

    def val_dataloader(self):
        if self.test_root is None:
            root = self.dataset_root
        else:
            root = self.test_root
        dataset = VOCDataset(dataset_root=root,
                             classes=open('classes.txt').read().splitlines(),
                             split='test')
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=yolo_det_collate)

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

    def _nms(self, pred):
        boxes, scores, idxs = pred
        new_boxes = []
        new_scores = []
        new_idxs = []
        for batch_idx, (box, score, idx) in enumerate(zip(boxes, scores,
                                                          idxs)):
            keep_idx = batched_nms(boxes=box,
                                   scores=score,
                                   idxs=idx,
                                   iou_threshold=0.6)
            new_boxes.append(box[keep_idx])
            new_scores.append(score[keep_idx])
            new_idxs.append(idx[keep_idx])
        return new_boxes, new_scores, new_idxs

    def _parse_preds(self, pred):
        boxes, scores, idxs = pred
        preds = []
        for box, score, idx in zip(boxes, scores, idxs):
            preds.append(dict(
                boxes=box,
                scores=score,
                labels=idx,
            ))
        return preds

    def _parse_targets(self, label, device=None):
        targets = []
        for batch_item in label:
            boxes = []
            labels = []
            for xcenter, ycenter, width, height, class_idx in batch_item:
                xmin, xmax = xcenter - (width / 2), xcenter + (width / 2)
                ymin, ymax = ycenter - (height / 2), ycenter + (height / 2)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_idx))
            boxes = torch.Tensor(boxes)
            labels = torch.Tensor(labels)
            if device is not None:
                boxes = boxes.to(device)
                labels = labels.to(device)
            targets.append(dict(
                boxes=boxes,
                labels=labels,
            ))

        return targets

    def training_step(self, batch, batch_idx):
        image, label = batch

        pred, loss = self.yolo_model(image, label)
        mean_ap = self.train_mean_ap(
            self._parse_preds(self._nms(pred)),
            self._parse_targets(label, image.device),
        )
        self.log('train/mean_ap', mean_ap)
        self.log('train/loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        train_mean_ap_epoch = self.train_mean_ap.compute()
        self.train_mean_ap.reset()

        self.log(
            'train/mean_ap_epoch',
            train_mean_ap_epoch,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred = self.yolo_model(image)
        self.val_mean_ap.update(
            self._parse_preds(self._nms(pred)),
            self._parse_targets(label, image.device),
        )

    def validation_epoch_end(self, outputs):
        val_mean_ap = self.val_mean_ap.compute()
        self.val_mean_ap.reset()

        self.log(
            'val/mean_ap',
            val_mean_ap,
            sync_dist=True,
        )


if __name__ == '__main__':
    from tqdm.auto import tqdm

    for split in ['train', 'val']:
        dataset = VOCDataset(
            dataset_root='/datasets/voc/VOC2012',
            classes=open('classes.txt').read().splitlines(),
            split=split,
        )

        for idx, (image, label) in enumerate(
                tqdm(dataset, desc="Sanity check {}".format(split))):
            assert image.shape[0] == 3 and \
                image.shape[1] == 416 and \
                image.shape[2] == 416

            assert label.shape[0] > 0 and \
                (0 <= label[:, :4]).all() and (label[:, :4] <= 1).all() and \
                    (label[:, 4].int() == label[:, 4]).all()

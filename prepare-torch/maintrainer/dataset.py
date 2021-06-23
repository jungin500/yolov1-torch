from glob import glob
from tqdm.auto import tqdm
from xml.etree import ElementTree

import torch
import os
import math

from PIL import Image
import cv2
import numpy as np

import albumentations as A

class VOCYOLOAnnotator(object):
    def __init__(self, annotation_root, image_root, model_cells=7, extension='jpg'):
        super().__init__()
        self.annotation_root = annotation_root
        self.image_root = image_root
        self.model_cells = model_cells

        self.annotation_files = glob(os.path.join(annotation_root, '*.xml'))
        self.labels = self.find_object_names()

    def find_object_names(self):
        object_map = {}
        for xml_filename in tqdm(self.annotation_files, desc='Annotation 내 Object Names 검색'):
            with open(xml_filename, 'r') as f:
                root = ElementTree.fromstring(f.read())
                for item in root.findall('object'):
                    object_name = item.find('name').text.strip()
                    object_map[object_name] = 1
        return list(sorted(object_map.keys()))

    def parse_annotation(self):
        annotations = []
        for xml_filename in tqdm(self.annotation_files, desc='Annotation 검색'):
            with open(xml_filename, 'r') as f:
                root = ElementTree.fromstring(f.read())
                size = root.find('size')

                filename = root.find('filename').text.strip()
                filepath = os.path.join(self.image_root, filename)
                image_width, image_height = int(size.find('width').text), int(size.find('height').text)
                objects = []
                for item in root.findall('object'):
                    object_name = item.find('name').text.strip()
                    object_id = self.labels.index(object_name)

                    object_bndbox = item.find('bndbox')
                    (xmin, ymin, xmax, ymax) = [
                        float(object_bndbox.find(key).text) for key in ['xmin', 'ymin', 'xmax', 'ymax']
                    ]

                    assert (object_id != -1)

                    xmin_norm, ymin_norm = xmin / image_width, ymin / image_height
                    width, height = xmax - xmin, ymax - ymin
                    width_norm, height_norm = width / image_width, height / image_height
                    xcenter_norm, ycenter_norm = xmin_norm + width_norm / 2, ymin_norm + height_norm / 2

                    # dynamic range tricks
                    # changes dynamic range from 0.0 to 7.0 and do floor()
                    # -> results 0, 1, 2, 3, 4, 5, 6!
                    cell_idx_x = int(xcenter_norm * self.model_cells)
                    cell_idx_y = int(ycenter_norm * self.model_cells)

                    cell_pos_x_norm = (xcenter_norm * self.model_cells - cell_idx_x)
                    cell_pos_y_norm = (ycenter_norm * self.model_cells - cell_idx_y)

                    objects.append(
                        [object_id, cell_idx_x, cell_idx_y, cell_pos_x_norm, cell_pos_y_norm, width_norm, height_norm])

                annotations.append((filepath, objects))

        return annotations


class VOCYolo(torch.utils.data.Dataset):
    def __init__(self, labels, annotations, boxes=2, grid_size=7, transform=None, augmentations=None):
        super(VOCYolo, self).__init__()
        self.labels = labels
        self.annotations = annotations
        self.transform = transform
        self.augmentations = None
        if augmentations is not None:
            self.augmentations = A.Compose(augmentations, bbox_params=A.BboxParams(
                format='albumentations', label_fields=['category_ids']
            ))
        self.boxes = boxes
        self.grid_size = grid_size
        self.classes = 20  # fixed as it's VOC dataset!
        self.one_hot = torch.eye(self.classes)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filepath, annotation = self.annotations[idx]

        if self.augmentations is not None:
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # x, y, width, height
            yolo_bboxes = [
                [(cell_idx_x + cell_pos_x) / 7, (cell_idx_y + cell_pos_y) / 7, width, height]
                for (class_id, cell_idx_x, cell_idx_y, cell_pos_x, cell_pos_y, width, height) in annotation
            ]
            # x_min, y_min, x_max, y_max
            bboxes = [
                [max(x - w / 2, 0), max(y - h / 2, 0), min(x + w / 2, 1), min(y + h / 2, 1)]
                for (x, y, w, h) in yolo_bboxes
            ]

            category_ids = [ class_id for (class_id, cell_idx_x, cell_idx_y, cell_pos_x, cell_pos_y, width, height) in annotation ]
            transform_result = self.augmentations(image=image, bboxes=bboxes, category_ids=category_ids)
            image_a, bboxes_a, categories_a = transform_result['image'], transform_result['bboxes'], transform_result['category_ids']
            image = Image.fromarray(image_a)
            # albumentations to yolo
            yolo_bboxes = [
                [xmin + ((xmax - xmin) / 2), ymin + ((ymax - ymin) / 2), xmax - xmin, ymax - ymin]
                for (xmin, ymin, xmax, ymax) in bboxes_a
            ]

            yolo_bboxes_annotation = [
                # cell_idx_x, cell_idx_y, cell_pos_x, cell_pos_y, width, height
                [min(int(x * 7), 6), min(int(y * 7), 6), x * 7 - int(x * 7), y * 7 - int(y * 7), w, h]
                for (x, y, w, h) in yolo_bboxes
            ]
            annotation = [[categories_a[i], *yolo_bboxes_annotation[i]] for i in range(len(yolo_bboxes_annotation))]
        else:
            image = Image.open(filepath).convert('RGB')  # case if image is grayscale

        # annotation = [[0, 3, 3, 0.5, 0.5, 0.95, 0.95] for _ in range(len(annotation))]

        # fill label
        label = torch.zeros((5 + self.classes, self.grid_size, self.grid_size), dtype=torch.float)
        for (class_id, cell_idx_x, cell_idx_y, cell_pos_x, cell_pos_y, width, height) in annotation:
            if label[4, cell_idx_y, cell_idx_x] != 1.0:
                label[:5, cell_idx_y, cell_idx_x] = torch.from_numpy(
                    np.array([cell_pos_x, cell_pos_y, width, height, 1.0], dtype=np.double))
                label[5:, cell_idx_y, cell_idx_x] = self.one_hot[class_id]

        if self.transform is not None:
            image = self.transform(image)
        return image, label
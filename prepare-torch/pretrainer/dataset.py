import os
from tqdm.auto import tqdm
from glob import glob
import torch
from PIL import Image
import numpy as np


class LabelReader(object):
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        if 'pretrained' in label_file_path:
            print("INFO: Using Pretrained label list! (not custom one)")

    def load_label(self):
        label_map = {}
        # Read label file into label map
        if os.path.isfile(self.label_file_path):
            with open(self.label_file_path, 'r') as f:
                label_name_body = f.read().strip()
                label_name_lines = label_name_body.split("\n")
                for label_entry in tqdm(label_name_lines, desc='레이블 파일 읽기 작업'):
                    synset_name, label_name = label_entry.strip().split("|")
                    label_map[synset_name] = label_name

            print(f"레이블 파일 읽기 완료: 총 {len(list(label_map.keys()))}개 레이블 검색")
            return label_map
        else:
            return None

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, labels, root_dir, transform=None, classes=1000):
        super(ImageNet, self).__init__()

        self.labels = labels
        self.transform = transform
        self.classes = classes

        self.img_path_list = []
        self.img_class_list = []
        self.load_list(root_dir)

    def load_list(self, root_dir):
        label_index = 0
        for label in tqdm(self.labels.keys(), desc='이미지 파일 리스트 읽기 작업'):
            if label_index >= self.classes:
                print("맨앞 20개의 레이블만 사용합니다.")
                break
            item_dir = os.path.join(root_dir, label)
            file_list = glob(item_dir + os.sep + "*.JPEG")
            self.img_path_list += file_list
            self.img_class_list += [label_index] * len(file_list)
            label_index += 1

        if len(self.img_path_list) != len(self.img_class_list):
            raise RuntimeError(f"이미지 데이터 {len(self.img_path_list)}개와 클래스 데이터 {len(self.img_class_list)}개가 서로 다릅니다!")

        print(f"총 {len(self.img_path_list)}개 이미지 리스트 데이터 및 실효 레이블 {len(list(set(self.img_class_list)))}개 로드 성공")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        # PIL-version
        image = Image.open(self.img_path_list[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.Tensor([self.img_class_list[idx]]).type(torch.int64).squeeze(dim=0)
        return image, label
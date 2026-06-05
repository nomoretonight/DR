import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mode, img_size):
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.GaussNoise(p=0.5),
            ], p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

class FundusDataset(Dataset):
    def __init__(self, df, image_root, img_size=512, mode='train', filter_gradable=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.img_size = img_size
        self.mode = mode
        self.filter_gradable = filter_gradable
        if filter_gradable and 'gradable' in self.df.columns:
            self.df = self.df[self.df['gradable'] == 1].reset_index(drop=True)
        self.transform = get_transforms(mode, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        image = np.array(Image.open(img_path).convert('RGB'))
        image = self.transform(image=image)['image']
        label = int(row['label'])            # 0-4
        referable = int(row['referable'])    # 0/1
        return image, label, referable, row.get('patient_id', idx)

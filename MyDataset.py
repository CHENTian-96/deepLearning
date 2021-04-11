# -*- coding: utf-8 -*- 
# @Time : 2021/4/4 15:09 
# @Author : CHENTian
# @File : MyDataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_txt, transform=None):
        self.root_dir = root_dir
        self.data_txt = data_txt
        self.transform = transform
        self.paths_labels = []
        with open(root_dir + data_txt, "r") as f:
            for img_info in f:
                img_info = img_info.rstrip()
                path_label = img_info.split()
                self.paths_labels.append((path_label[0], int(path_label[1])))

    def __getitem__(self, index):
        path, label = self.paths_labels[index]
        image = Image.open(self.root_dir + path).convert("RGB")
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.paths_labels)


# train_data = MyDataset("./data/", "train.txt")
# train_loder = data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)












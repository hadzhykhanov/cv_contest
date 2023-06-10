import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data.dataset import Dataset


class RotationDataset(Dataset):
    def __init__(self, file_list, transform, target_list=None):
        self.file_list = file_list
        self.transform = transform
        self.target_list = target_list

    def __getitem__(self, idx):
        file, target = self.file_list[idx], self.target_list[idx]
        img = Image.open(file).convert("RGB")
        img = np.array(img)

        if target == 1:
            img = np.flip(img, axis=1)

        # if self.target_list is None:
        #     target = 0
        #     if random.randint(0, 1) == 1:
        #         target = 1
        # else:
        #     target = self.target_list[idx]
        #
        # if target == 1:
        #     img = np.flip(img, axis=1)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = torch.tensor(img)
        img = img.permute(2, 0, 1).float()

        return {"image": img, "target": target}

    def __len__(self):
        return len(self.file_list)

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class OCRDataset(Dataset):
    def __init__(self, file_list, targets_encoded, transform):
        self.file_list = file_list
        self.targets_encoded = targets_encoded
        self.transform = transform

    def __getitem__(self, idx):
        file, target = self.file_list[idx], self.targets_encoded[idx]
        img = Image.open(file).convert("RGB")
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = torch.tensor(img)
        img = img.permute(2, 0, 1).float()

        return {"image": img, "target": target}

    def __len__(self):
        return len(self.file_list)


# class ValOCRDataset(Dataset):
#     def __init__(self, file_list, transform):
#         self.file_list = file_list
#         self.transform = transform
#
#     def __getitem__(self, index):
#         file = self.file_list[index]
#         img = Image.open(file).convert("RGB")
#         img = np.array(img)
#
#         if self.transform is not None:
#             img = self.transform(image=img)["image"]
#
#         return {
#             "image": img,
#             "file": os.path.basename(file),
#         }
#
#     def __len__(self):
#         return len(self.file_list)

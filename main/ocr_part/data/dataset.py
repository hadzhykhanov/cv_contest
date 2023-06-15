import os.path
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class OCRDataset(Dataset):
    def __init__(
        self,
        file_list,
        targets_encoded,
        rotation_labels,
        idx_to_angle,
        transform,
    ):
        self.file_list = file_list
        self.targets_encoded = targets_encoded
        self.rotation_labels = rotation_labels
        self.idx_to_angle = idx_to_angle
        self.transform = transform

    @staticmethod
    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Получаем новые размеры изображения после поворота
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Меняем центр изображения для его поворота
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        # Вычисляем средний цвет краев изображения
        border_color = [
            np.mean(image[0, :], axis=0),
            np.mean(image[-1, :], axis=0),
            np.mean(image[:, 0], axis=0),
            np.mean(image[:, -1], axis=0),
        ]
        border_color = np.mean(border_color, axis=0).astype(int)

        rotated = cv2.warpAffine(
            image,
            M,
            (nW, nH),
            borderMode=cv2.INTER_LINEAR,
            borderValue=tuple(border_color.tolist()),
        )

        return rotated

    def __getitem__(self, idx):
        file = self.file_list[idx]
        target = self.targets_encoded[idx] if self.targets_encoded else None
        rotation_label = self.rotation_labels[os.path.basename(file)]

        # print(
        #     file,
        #     target,
        #     rotation_label,
        # )

        for _ in range(15):
            print("LOL")

        img = cv2.imread(file)
        img = self.rotate_image(image=img, angle=self.idx_to_angle[rotation_label])

        # img_height, img_width = img.shape[0], img.shape[1]
        # if img_height > img_width:
        #     img = np.transpose(img, (1, 0, 2))
        #     img = np.flip(img, axis=1)

        # img = torch.from_numpy(img).permute(1, 0, 2).flip(1).numpy()

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = torch.tensor(img)
        img = img.permute(2, 0, 1).float()

        out_dict = {"image": img}
        if self.targets_encoded:
            out_dict["target"] = target

        return out_dict

    def __len__(self):
        return len(self.file_list)

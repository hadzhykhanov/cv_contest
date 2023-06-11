import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import albumentations
import os
from sklearn.model_selection import train_test_split
from data.dataset import RotationDataset


def read_data(
    input_train_folder_path: str,
    input_test_folder_path: str,
    input_ocr_targets_path: str,
    id_column_name: str,
    rotation_labels_path: str,
):
    input_train_folder_path, input_test_folder_path = (
        pathlib.Path(input_train_folder_path),
        pathlib.Path(input_test_folder_path),
    )

    targets_df = pd.read_csv(input_ocr_targets_path)[id_column_name]
    targets_df = set(targets_df.values)
    train_files = [str(path) for path in input_train_folder_path.glob("*.jpg")]
    out_list = []

    for path in train_files:
        if os.path.basename(path) in targets_df:
            img = Image.open(path).convert("RGB")
            img_arr = np.array(img)

            if img_arr.shape[0] * 3 <= img_arr.shape[1]:
                out_list.append(path)

    for local_path in input_test_folder_path.glob("*.jpg"):
        img = Image.open(local_path).convert("RGB")
        img_arr = np.array(img)

        if img_arr.shape[0] * 3 <= img_arr.shape[1]:
            out_list.append(local_path)

    labels = pd.read_csv(rotation_labels_path).values.ravel()

    return out_list, labels


def split_train_test_data(file_list, labels, test_size, random_state):
    (train_files, test_files, train_labels, test_labels) = train_test_split(
        file_list,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    return train_files, test_files, train_labels, test_labels


def make_loaders(
    train_files,
    train_labels,
    train_batch_size,
    test_files,
    test_labels,
    test_batch_size,
):
    # print(
    #     type(train_labels[:5]),
    #     type(test_labels[:5]),
    # )
    train_transform = albumentations.Compose(
        [
            albumentations.Resize(64, 320),
        ]
    )

    train_dataset = RotationDataset(
        file_list=train_files,
        target_list=train_labels,
        transform=train_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True
    )

    test_transform = albumentations.Compose([albumentations.Resize(64, 320)])
    test_dataset = RotationDataset(
        file_list=test_files,
        target_list=test_labels,
        transform=test_transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader

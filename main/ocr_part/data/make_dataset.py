import pathlib
import os
import torch
import pandas as pd
import albumentations
from data.dataset import OCRDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def read_data(
    input_train_data_path,
    input_val_data_path,
    input_targets_path,
    input_rotation_train_path,
    input_rotation_val_path,
):
    train_files_path, val_files_path = pathlib.Path(
        input_train_data_path
    ), pathlib.Path(input_val_data_path)

    targets_df = pd.read_csv(input_targets_path)
    targets_dct = dict(
        zip(targets_df.iloc[:, 0].values.ravel(), targets_df.iloc[:, 1].values.ravel())
    )

    print(targets_dct.items()[:5])

    train_files = [str(path) for path in train_files_path.glob("*.jpg")]
    train_files = list(
        filter(lambda x: os.path.basename(x) in targets_dct, train_files)
    )
    val_files = [str(path) for path in val_files_path.glob("*jpg")]

    targets_orig = list()
    targets_splitted = list()
    targets_flattened = set()

    for file in train_files:
        file_basename = os.path.basename(file)

        if file_basename in targets_dct:
            target = targets_dct[file_basename]

            targets_orig.append(target)
            targets_splitted.append(list(target))
            targets_flattened.update(target)

    rotations_train = pd.read_csv(input_rotation_train_path)
    rotations_test = pd.read_csv(input_rotation_val_path)

    rotations_train = dict(
        zip(
            rotations_train.iloc[:, 0].values.ravel(),
            rotations_train.iloc[:, 1].values.ravel(),
        )
    )

    rotations_test = dict(
        zip(
            rotations_test.iloc[:, 0].values.ravel(),
            rotations_test.iloc[:, 1].values.ravel(),
        )
    )

    print(rotations_train.items()[:5])
    print(rotations_test.items()[:5])

    return (
        train_files,
        val_files,
        targets_orig,
        targets_splitted,
        list(targets_flattened),
        rotations_train,
        rotations_test,
    )


def split_train_test_data(
    file_list, targets_encoded, targets_orig, test_size, random_state
):
    (
        train_files,
        test_files,
        train_encoded_targets,
        test_encoded_targets,
        _,
        test_orig_targets,
    ) = train_test_split(
        file_list,
        targets_encoded,
        targets_orig,
        test_size=test_size,
        random_state=random_state,
    )

    return (
        train_files,
        test_files,
        train_encoded_targets,
        test_encoded_targets,
        _,
        test_orig_targets,
    )


def collate_fn(batch):
    images, seqs, seq_lens = [], [], []

    for dct in batch:
        images.append(dct["image"])
        seqs.extend(dct["target"])
        seq_lens.append(len(dct["target"]))

    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()

    batch = {"image": images, "seq": seqs, "seq_len": seq_lens}
    return batch


def make_loaders(
    train_files,
    train_encoded_targets,
    train_batch_size,
    rotations_train,
    test_files,
    test_encoded_targets,
    test_batch_size,
    rotations_test,
    val_files,
    val_batch_size,
    rotations_val,
    resize,
    num_workers,
    idx_to_angle,
):
    train_transform = albumentations.Compose(
        [
            albumentations.Resize(*resize),
        ]
    )

    train_dataset = OCRDataset(
        file_list=train_files,
        targets_encoded=train_encoded_targets,
        transform=train_transform,
        rotation_labels=rotations_train,
        idx_to_angle=idx_to_angle,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_transform = albumentations.Compose([albumentations.Resize(*resize)])

    test_dataset = OCRDataset(
        file_list=test_files,
        targets_encoded=test_encoded_targets,
        transform=test_transform,
        rotation_labels=rotations_test,
        idx_to_angle=idx_to_angle,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_transform = albumentations.Compose(
        [
            albumentations.Resize(*resize),
        ]
    )

    val_dataset = OCRDataset(
        file_list=val_files,
        targets_encoded=None,
        transform=val_transform,
        rotation_labels=rotations_val,
        idx_to_angle=idx_to_angle,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, val_loader

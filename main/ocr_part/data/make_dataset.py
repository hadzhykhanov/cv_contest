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
    id_column_name,
    target_column_name,
    train_rotation_scoring_path,
    test_rotation_scoring_path,
):
    train_files_path, val_files_path = pathlib.Path(
        input_train_data_path
    ), pathlib.Path(input_val_data_path)

    targets_df = pd.read_csv(input_targets_path)
    targets_dct = dict(zip(targets_df[id_column_name], targets_df[target_column_name]))

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

    train_rotation_scoring_df = pd.read_csv(train_rotation_scoring_path)
    test_rotation_scoring_df = pd.read_csv(test_rotation_scoring_path)
    train_rotation_scoring_dct = dict(
        zip(train_rotation_scoring_df["id"], train_rotation_scoring_df["label"])
    )

    test_rotation_scoring_dct = dict(
        zip(train_rotation_scoring_df["id"], train_rotation_scoring_df["label"])
    )

    return (
        train_files,
        val_files,
        targets_orig,
        targets_splitted,
        list(targets_flattened),
        train_rotation_scoring_dct,
        test_rotation_scoring_dct,
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
    train_rotation_scoring_dct,
    test_files,
    test_encoded_targets,
    test_batch_size,
    test_rotation_scoring_dct,
):
    train_transform = albumentations.Compose(
        [
            albumentations.Resize(64, 320),
        ]
    )

    train_dataset = OCRDataset(
        file_list=train_files,
        targets_encoded=train_encoded_targets,
        transform=train_transform,
        train_rotation_scoring_dct=train_rotation_scoring_dct,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_transform = albumentations.Compose([albumentations.Resize(64, 320)])

    test_dataset = OCRDataset(
        file_list=test_files,
        targets_encoded=test_encoded_targets,
        transform=test_transform,
        test_rotation_scoring_dct=test_rotation_scoring_dct,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader

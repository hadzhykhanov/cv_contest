import pathlib
import albumentations
from data.dataset import RotationDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def read_data(input_data_path):
    files_path = pathlib.Path(input_data_path)

    files = [str(path) for path in files_path.glob("*")]

    return files


def split_train_test_data(file_list, targets, test_size, random_state):
    (
        train_files,
        test_files,
        train_targets,
        test_targets,
    ) = train_test_split(
        file_list,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    return (
        train_files,
        test_files,
        train_targets,
        test_targets,
    )


# def collate_fn(batch):
#     images, seqs, seq_lens = [], [], []
#
#     for dct in batch:
#         images.append(dct["image"])
#         seqs.extend(dct["target"])
#         seq_lens.append(len(dct["target"]))
#
#     images = torch.stack(images)
#     seqs = torch.Tensor(seqs).int()
#     seq_lens = torch.Tensor(seq_lens).int()
#
#     batch = {"image": images, "seq": seqs, "seq_len": seq_lens}
#     return batch


def make_loaders(
    train_files,
    train_targets,
    train_batch_size,
    test_files,
    test_targets,
    test_batch_size,
    resize,
    num_workers,
    idx_to_angle,
):
    train_transform = albumentations.Compose(
        [
            albumentations.Resize(*resize),
        ]
    )

    train_dataset = RotationDataset(
        file_list=train_files,
        targets=train_targets,
        transform=train_transform,
        idx_to_angle=idx_to_angle,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_transform = albumentations.Compose(
        [
            albumentations.Resize(*resize),
        ]
    )

    test_dataset = RotationDataset(
        file_list=test_files,
        targets=test_targets,
        transform=test_transform,
        idx_to_angle=idx_to_angle,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader

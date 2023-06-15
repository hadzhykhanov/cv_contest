import os
import hydra
import albumentations
from data.dataset import RotationDataset
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from data.make_dataset import read_data
from omegaconf import DictConfig
from model.model import ResNet18
from entities import entities
from model.model_fit_predict import (
    inference,
    save_predictions,
    load_model,
)


def get_path(filename):
    return os.path.join(os.getcwd(), filename)


config_store = ConfigStore.instance()
config_store.store(name="config", node=entities.Config)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="rotation_config"
)
def run_training(config: DictConfig):
    inference_files = read_data(input_data_path=config.data_params.input_data_path)

    train_transform = albumentations.Compose(
        [
            albumentations.Resize(*config.inference.resize),
        ]
    )

    inference_dataset = RotationDataset(
        file_list=inference_files,
        transform=train_transform,
        targets=None,
        idx_to_angle=None,
    )

    inference_loader = DataLoader(
        dataset=inference_dataset,
        batch_size=config.inference.inference_batch_size,
        shuffle=False,
        num_workers=config.inference.num_workers,
    )

    model = ResNet18(num_classes=config.model_params.num_classes).to(
        config.training_params.device
    )

    model = load_model(
        model=model,
        input_model_path=config.inference.input_model_path,
    )

    inference_preds = inference(
        model=model,
        inference_loader=inference_loader,
        device=config.inference.device,
    )

    save_predictions(
        paths=list(map(lambda x: os.path.basename(x), inference_files)),
        preds=inference_preds,
        targets=None,
        output_predictions_path=config.data_params.output_predictions_path,
    )


if __name__ == "__main__":
    run_training()

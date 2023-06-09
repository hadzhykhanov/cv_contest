import os
import torch
import hydra
from hydra.core.config_store import ConfigStore
from data.make_dataset import read_data, split_train_test_data, make_loaders
from omegaconf import DictConfig
from model.model import ResNet18
from entities import entities
from sklearn.metrics import accuracy_score
from pprint import pprint
from model.model_fit_predict import (
    train_model,
    evaluate_model,
    save_model,
    save_metrics,
    save_predictions,
)


def get_path(filename):
    return os.path.join(os.getcwd(), filename)


config_store = ConfigStore.instance()
config_store.store(name="config", node=entities.Config)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="rotation_config"
)
def run_training(config: DictConfig):
    files = read_data(input_data_path=config.train.data_params.input_data_path)
    targets = [int(path.split("_")[-1].split(".")[0]) for path in files]
    targets = list(map(lambda x: config.train.data_params.angle_to_idx[x], targets))

    (
        train_files,
        test_files,
        train_targets,
        test_targets,
    ) = split_train_test_data(
        file_list=files,
        targets=targets,
        test_size=config.train.data_params.test_size,
        random_state=config.train.data_params.random_state,
    )

    train_loader, test_loader = make_loaders(
        train_files=train_files,
        train_targets=train_targets,
        train_batch_size=config.train.training_params.train_batch_size,
        test_files=test_files,
        test_targets=test_targets,
        test_batch_size=config.train.training_params.test_batch_size,
        resize=config.train.aug_params.resize,
        num_workers=config.train.training_params.num_workers,
        idx_to_angle=dict(
            zip(
                config.train.data_params.angle_to_idx.values(),
                config.train.data_params.angle_to_idx.keys(),
            )
        ),
    )

    model = ResNet18(num_classes=config.train.model_params.num_classes).to(
        config.train.training_params.device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.training_params.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.train.training_params.factor,
        patience=config.train.training_params.patience,
        verbose=True,
    )

    for epoch in range(1, config.train.training_params.epochs_num + 1):
        train_loss = train_model(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=config.train.training_params.device,
        )

        test_preds, test_loss = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=config.train.training_params.device,
        )

        test_accuracy = accuracy_score(test_targets, test_preds)

        # print(
        # list(filter(lambda x: x[1] != x[2], zip(files, test_targets, test_preds)))
        # )

        for value in config.train.data_params.angle_to_idx.values():
            iterator = zip(test_targets, test_preds)
            lst = list(filter(lambda x: x[0] == value, iterator))
            true = list(filter(lambda x: x[0] == x[1], lst))

            print(f"{value}, {len(true) / len(lst)}")

        pprint(
            list(
                zip(
                    test_preds,
                    test_targets,
                )
            )[:5]
        )
        print(f"{epoch=}, {train_loss=}, {test_loss=}, {test_accuracy=}")

        scheduler.step(test_loss)

    metrics = {"test_loss": test_loss, "accuracy": test_accuracy}
    save_model(
        model=model, output_model_path=config.train.data_params.output_model_path
    )
    save_metrics(
        metrics=metrics,
        output_metrics_path=get_path(config.train.data_params.output_metrics_path),
    )
    save_predictions(
        paths=list(
            map(
                lambda x: os.path.basename(x),
                test_files,
            )
        ),
        targets=test_targets,
        preds=test_preds,
        output_predictions_path=config.train.data_params.output_predictions_path,
    )


if __name__ == "__main__":
    run_training()

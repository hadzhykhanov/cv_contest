import os
import torch
import hydra
from hydra.core.config_store import ConfigStore
from data.make_dataset import read_data, split_train_test_data, make_loaders
from omegaconf import DictConfig
from model.model import ResNet18
from sklearn.metrics import accuracy_score, f1_score
from entities import entities
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
    file_list, labels = read_data(
        input_train_folder_path=config.data_params.input_train_folder_path,
        input_test_folder_path=config.data_params.input_test_folder_path,
        input_ocr_targets_path=config.data_params.input_ocr_targets_path,
        id_column_name=config.data_params.id_column_name,
        rotation_labels_path=config.data_params.rotation_labels_path,
    )

    train_files, test_files, train_labels, test_labels = split_train_test_data(
        file_list=file_list,
        labels=labels,
        test_size=config.data_params.test_size,
        random_state=config.data_params.random_state,
    )

    train_loader, test_loader = make_loaders(
        train_files=train_files,
        train_labels=train_labels,
        train_batch_size=config.training_params.train_batch_size,
        test_files=test_files,
        test_labels=test_labels,
        test_batch_size=config.training_params.test_batch_size,
    )

    model = ResNet18(num_classes=config.training_params.num_classes).to(
        config.training_params.device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training_params.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.training_params.factor,
        patience=config.training_params.patience,
        verbose=True,
    )

    for epoch in range(1, config.training_params.epochs_num + 1):
        # train_loss = train_model(
        #     model=model,
        #     data_loader=train_loader,
        #     optimizer=optimizer,
        #     device=config.training_params.device,
        # )

        test_preds, test_loss = evaluate_model(
            model=model, data_loader=test_loader, device=config.training_params.device
        )

        accuracy = accuracy_score(test_labels, test_preds.cpu().numpy())
        f1 = f1_score(test_labels, test_preds.cpu().numpy())

        # print(f"{epoch=}, {train_loss=}, {test_loss=}, {accuracy=}, {f1=}")
        print(f"{epoch=}, {test_loss=}, {accuracy=}, {f1=}")

        scheduler.step(test_loss)

    metrics = {"test_loss": test_loss, "accuracy": accuracy, "f1_score": f1}
    save_model(model=model, output_model_path=config.data_params.output_model_path)
    save_metrics(
        metrics=metrics,
        output_metrics_path=get_path(config.data_params.output_metrics_path),
    )
    save_predictions(
        test_labels=test_labels,
        test_preds=test_preds,
        output_predictions_path=config.data_params.output_predictions_path,
    )


if __name__ == "__main__":
    run_training()

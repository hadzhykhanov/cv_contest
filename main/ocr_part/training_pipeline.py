import os
import torch
import hydra
import torchmetrics
from hydra.core.config_store import ConfigStore
from data.make_dataset import read_data, split_train_test_data, make_loaders
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from model.predictions_preprocessing import preprocess_prediction, decode_predictions
from model.model import CRNN
from joblib import dump
from entities import entities
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
    version_base=None, config_path="../../configs/ocr_part", config_name="config"
)
def run_training(config: DictConfig):
    (
        train_files,
        val_files,
        targets_orig,
        targets_splitted,
        targets_flattened,
    ) = read_data(
        input_train_data_path=config.data_params.input_train_data_path,
        input_val_data_path=config.data_params.input_val_data_path,
        input_targets_path=config.data_params.input_targets_path,
        id_column_name=config.data_params.id_column_name,
        target_column_name=config.data_params.target_column_name,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(targets_flattened)
    dump(label_encoder, config.data_params.output_encoder_path)
    targets_encoded = [
        [val + 1 for val in label_encoder.transform(target)]
        for target in targets_splitted
    ]

    (
        train_files,
        test_files,
        train_encoded_targets,
        test_encoded_targets,
        _,
        test_orig_targets,
    ) = split_train_test_data(
        file_list=train_files,
        targets_encoded=targets_encoded,
        targets_orig=targets_orig,
        test_size=config.data_params.test_size,
        random_state=config.data_params.random_state,
    )

    train_loader, test_loader = make_loaders(
        train_files=train_files,
        train_encoded_targets=train_encoded_targets,
        train_batch_size=config.training_params.train_batch_size,
        test_files=test_files,
        test_encoded_targets=test_encoded_targets,
        test_batch_size=config.training_params.test_batch_size,
    )

    # for data in train_loader:
    #     print(**data)

    model = CRNN(num_chars=len(label_encoder.classes_)).to(
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
        train_loss = train_model(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=config.training_params.device,
        )

        test_preds, test_loss = evaluate_model(
            model=model, data_loader=test_loader, device=config.training_params.device
        )

        test_decoded_preds = [
            decode_predictions(preds=prediction, encoder=label_encoder)
            for prediction in test_preds
        ]

        test_decoded_preds = [
            preprocess_prediction(prediction)
            for lst in test_decoded_preds
            for prediction in lst
        ]

        char_error_rate = torchmetrics.functional.char_error_rate(
            preds=test_decoded_preds, target=test_orig_targets
        ).item()

        pprint(
            list(
                zip(
                    test_orig_targets,
                    test_decoded_preds,
                )
            )[:6]
        )
        print(f"{epoch=}, {train_loss=}, {test_loss=}, {char_error_rate=}")

        scheduler.step(test_loss)

    metrics = {"test_loss": test_loss, "char_error_rate": char_error_rate}
    save_model(model=model, output_model_path=config.data_params.output_model_path)
    save_metrics(
        metrics=metrics,
        output_metrics_path=get_path(config.data_params.output_metrics_path),
    )
    save_predictions(
        test_orig_targets=test_orig_targets,
        test_decoded_preds=test_decoded_preds,
        output_predictions_path=config.data_params.output_predictions_path,
    )


if __name__ == "__main__":
    run_training()

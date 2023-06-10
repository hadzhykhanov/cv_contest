from dataclasses import dataclass


@dataclass()
class GeneralConfig:
    random_state: int


@dataclass()
class DataConfig:
    input_train_folder_path: str
    input_test_folder_path: str
    output_model_path: str
    output_metrics_path: str
    output_predictions_path: str
    id_column_name: str
    input_ocr_targets_path: str
    rotation_labels_path: str
    test_size: float
    random_state: int


@dataclass()
class TrainingConfig:
    num_classes: int
    epochs_num: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    factor: float
    patience: int
    device: str


@dataclass()
class Config:
    general: GeneralConfig
    data_params: DataConfig
    training_params: TrainingConfig

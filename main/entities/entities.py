from dataclasses import dataclass


@dataclass()
class GeneralConfig:
    random_state: int


@dataclass()
class DataConfig:
    input_train_data_path: str
    input_val_data_path: str
    input_targets_path: str
    id_column_name: str
    target_column_name: str
    output_model_path: str
    output_metrics_path: str
    output_predictions_path: str
    test_size: float
    random_state: int


@dataclass()
class SplittingConfig:
    test_size: float
    random_state: int


@dataclass()
class TrainingConfig:
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
    splitting_params: SplittingConfig
    training_params: TrainingConfig

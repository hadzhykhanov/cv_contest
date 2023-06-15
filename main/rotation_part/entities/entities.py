from dataclasses import dataclass


@dataclass()
class GeneralConfig:
    random_state: int


@dataclass()
class DataConfig:
    input_data_path: str
    output_model_path: str
    output_metrics_path: str
    output_predictions_path: str
    angle_to_idx: dict
    test_size: float
    random_state: int


@dataclass()
class AugConfig:
    resize: list


@dataclass()
class ModelConfig:
    num_classes: int


@dataclass()
class TrainingConfig:
    epochs_num: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    factor: float
    patience: int
    device: str
    num_workers: int


@dataclass()
class TrainConfig:
    general: GeneralConfig
    data_params: DataConfig
    aug_params: AugConfig
    model_params: ModelConfig
    training_params: TrainingConfig


@dataclass()
class InferenceConfig:
    resize: list
    inference_batch_size: int
    num_workers: int
    input_model_path: str
    device: str


@dataclass()
class Config:
    train: TrainConfig
    inference: InferenceConfig

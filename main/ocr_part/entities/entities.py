from dataclasses import dataclass


@dataclass()
class GeneralConfig:
    random_state: int


@dataclass()
class DataConfig:
    input_train_data_path: str
    input_val_data_path: str
    input_targets_path: str
    input_rotation_train_path: str
    input_rotation_test_path: str
    output_model_path: str
    output_test_metrics_path: str
    output_test_predictions_path: str
    output_val_predictions_path: str
    output_encoder_path: str
    test_size: float
    random_state: int


@dataclass()
class ModelConfig:
    resize: list
    cnn_output_len: int
    rnn_hidden_size: int
    rnn_num_layers: int
    rnn_dropout: float
    rnn_bidirectional: bool


@dataclass()
class TrainingConfig:
    epochs_num: int
    train_batch_size: int
    test_batch_size: int
    val_batch_size: int
    learning_rate: float
    factor: float
    patience: int
    device: str
    num_workers: int


@dataclass()
class Config:
    general: GeneralConfig
    data_params: DataConfig
    model_params: ModelConfig
    training_params: TrainingConfig

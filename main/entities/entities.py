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
    output_encoder_path: str
    test_size: float
    random_state: int


@dataclass()
class ModelConfig:
    cnn_input_height: int
    cnn_input_width: int
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
    learning_rate: float
    factor: float
    patience: int
    device: str
    num_workers: 2


@dataclass()
class Config:
    general: GeneralConfig
    data_params: DataConfig
    model_params: ModelConfig
    training_params: TrainingConfig

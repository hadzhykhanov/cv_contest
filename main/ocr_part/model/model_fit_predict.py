import json
import torch
import pandas as pd
from tqdm import tqdm


def train_model(model, data_loader, optimizer, device):
    model.train()
    train_loss = 0

    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for key, value in data.items():
            data[key] = value.to(device)

        optimizer.zero_grad()
        _, loss = model(data["image"], data["seq"], data["seq_len"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    test_loss = 0
    test_preds = []

    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for key, value in data.items():
            data[key] = value.to(device)

        batch_preds, loss = model(data["image"], data["seq"], data["seq_len"])
        test_loss += loss.item()
        test_preds.append(batch_preds)

    return test_preds, test_loss / len(data_loader)


@torch.no_grad()
def validate_model(model, inference_loader, device):
    model.eval()
    validation_preds = []

    tk = tqdm(inference_loader, total=len(inference_loader))
    for data in tk:
        for key, value in data.items():
            data[key] = value.to(device)

        batch_preds, _ = model(data["image"], None)
        validation_preds.append(batch_preds)

    return validation_preds


def save_model(model, output_model_path):
    torch.save(model.state_dict(), output_model_path)


def save_metrics(metrics, output_metrics_path):
    with open(output_metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)


def save_predictions(paths, targets, preds, output_predictions_path):
    df_dict = {
        "Id": paths,
        "Predicted": preds,
    }

    if targets:
        df_dict["target"] = targets

    df = pd.DataFrame(df_dict)

    df.to_csv(output_predictions_path, index=False)

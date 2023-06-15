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
        _, loss = model(data["image"], data["target"])
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

        batch_preds, loss = model(data["image"], data["target"])
        test_loss += loss.item()
        test_preds.extend(batch_preds.argmax(dim=1).cpu().numpy())

    return test_preds, test_loss / len(data_loader)


@torch.no_grad()
def inference(model, data_loader, device):
    model.eval()
    inference_preds = []

    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for key, value in data.items():
            data[key] = value.to(device)

        batch_preds, _ = model(data["image"], data["target"])
        inference_preds.extend(batch_preds.argmax(dim=1).cpu().numpy())

    return inference_preds


def save_model(model, output_model_path):
    torch.save(model.state_dict(), output_model_path)


def load_model(model, input_model_path):
    model.load_state_dict(torch.load(input_model_path))
    return model


def save_metrics(metrics, output_metrics_path):
    with open(output_metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)


def save_predictions(paths, targets, preds, output_predictions_path):
    df_dict = {
        "path": paths,
        "prediction": preds,
    }

    if targets is not None:
        df_dict["target"] = targets

    df_dict = pd.DataFrame(df_dict)
    df_dict.to_csv(output_predictions_path, index=False)

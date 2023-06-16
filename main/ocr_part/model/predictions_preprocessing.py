import torch


def preprocess_prediction(prediction, special_symbol="-"):
    current = ""
    for char in prediction:
        if current == "":
            current = char
        else:
            if char != current[-1]:
                current += char

    output = ""
    for char in current:
        if char != special_symbol:
            output += char

    return output


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, dim=2)
    preds = torch.argmax(preds, dim=2)
    preds = preds.detach().cpu().numpy()

    # print(preds.shape)

    cap_preds = []
    for counter in range(preds.shape[0]):
        temp = []
        for k in preds[counter, :]:
            k = k - 1
            if k == -1:
                temp.append("-")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)

    return cap_preds

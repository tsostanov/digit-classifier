"""Метрики через матрицу ошибок: строки — истина, столбцы — прогноз."""

import numpy as np
import torch
from torch import nn


def classification_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = matrix.diagonal()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator != 0)
    return {
        "accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
        "samples": int(matrix.sum()),
    }


@torch.inference_mode()
def evaluate(model, loader, device, max_errors=16):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    matrix = np.zeros((10, 10), dtype=np.int64)
    total_loss, offset, errors = 0.0, 0, []
    predictions = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        predicted = logits.argmax(1)
        truth, guess = labels.cpu().numpy(), predicted.cpu().numpy()
        np.add.at(matrix, (truth, guess), 1)
        predictions.extend(guess.tolist())
        for index in np.flatnonzero(truth != guess):
            if len(errors) >= max_errors:
                break
            errors.append({
                "index": offset + int(index), "true": int(truth[index]),
                "predicted": int(guess[index]), "image": images[index].cpu(),
            })
        offset += len(labels)
    if offset == 0:
        raise ValueError("Невозможно оценить пустой набор.")
    report = classification_metrics(matrix)
    report["loss"] = total_loss / offset
    return report, errors, predictions

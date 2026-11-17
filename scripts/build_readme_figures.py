"""Иллюстрации README из сохранённых предсказаний и опубликованных весов."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from digit_classifier.data import CustomDigits, mnist_data
from digit_classifier.engine import file_hash, load_checkpoint, seed_everything
from digit_classifier.images import MEAN, STD


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_predictions(dataset, directory):
    predictions = read_json(directory / "predictions.json")
    report = read_json(directory / "metrics.json")
    labels = [int(dataset[index][1]) for index in range(len(dataset))]
    assert len(predictions) == len(labels) == report["samples"]
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    assert matrix.tolist() == report["confusion_matrix"]
    return predictions, [index for index, (a, b) in enumerate(zip(labels, predictions)) if a != b]


def probabilities_for(model, dataset, indices, predictions):
    with torch.inference_mode():
        probabilities = model(torch.stack([dataset[index][0] for index in indices])).softmax(1)
    assert probabilities.argmax(1).tolist() == [predictions[index] for index in indices]
    return probabilities.max(1).values.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/lenet_sgd"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--custom-dir", type=Path, default=Path("custom_test"))
    args = parser.parse_args()
    root = args.results
    seed_everything(42, threads=4)
    model, checkpoint = load_checkpoint(root / "best.pt", "cpu")
    digest = file_hash(root / "best.pt")
    assert digest == read_json(root / "summary.json")["checkpoint_sha256"]
    assert file_hash(root / "split.json") == checkpoint["split_sha256"]
    split = read_json(root / "split.json")
    _, validation, _ = mnist_data(args.data_dir, root / "split.json", checkpoint["config"]["seed"])
    predictions, errors = checked_predictions(validation, root / "validation")
    output = root / "readme"
    output.mkdir(parents=True, exist_ok=True)
    selected = errors[:48]
    scores = probabilities_for(model, validation, selected, predictions)
    metadata = {"checkpoint_sha256": digest, "validation_errors_total": len(errors),
                "selection": "first_48_in_validation_order", "validation_examples": []}
    for page, start in enumerate(range(0, len(selected), 16), 1):
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for axis in axes.flat:
            axis.axis("off")
        for axis, offset in zip(axes.flat, range(start, min(start + 16, len(selected)))):
            index = selected[offset]
            image, label = validation[index]
            axis.imshow(image.squeeze().numpy() * STD + MEAN, cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"Верно: {label} | Ответ: {predictions[index]}\n"
                           f"№{index}, оценка {scores[offset]:.0%}", fontsize=10)
            metadata["validation_examples"].append({"index": index,
                "mnist_index": split["validation"][index],
                "true": int(label), "predicted": predictions[index], "score": scores[offset]})
        fig.tight_layout(pad=1.2)
        fig.savefig(output / f"validation_errors_{page:02}.png", dpi=140)
        plt.close(fig)

    custom = CustomDigits(args.custom_dir)
    provenance = read_json(root / "custom/evaluation_config.json")
    assert provenance["checkpoint_sha256"] == digest
    assert file_hash(args.custom_dir / "labels.csv") == provenance["manifest_sha256"]
    for record in provenance["images"]:
        assert file_hash(args.custom_dir / record["path"]) == record["sha256"]
    guesses, mistakes = checked_predictions(custom, root / "custom")
    metadata["custom_errors_total"] = len(mistakes)
    metadata["custom_examples"] = []
    if mistakes:
        scores = probabilities_for(model, custom, mistakes, guesses)
        fig, axes = plt.subplots(2, len(mistakes), figsize=(2.3 * len(mistakes), 4.8), squeeze=False)
        for column, (index, score) in enumerate(zip(mistakes, scores)):
            path, label = custom.records[index]
            with Image.open(path) as source:
                axes[0, column].imshow(source.convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[0, column].set_title(f"Рисунок №{index}\nМетка: {label}", fontsize=11)
            tensor, _ = custom[index]
            axes[1, column].imshow(tensor.squeeze().numpy() * STD + MEAN, cmap="gray", vmin=0, vmax=1)
            axes[1, column].set_title(f"Вход сети 28×28\nОтвет: {guesses[index]} ({score:.0%})", fontsize=11)
            metadata["custom_examples"].append({"index": index, "true": label,
                "predicted": guesses[index], "score": score})
        for axis in axes.flat:
            axis.axis("off")
        fig.tight_layout(pad=1.2)
        fig.savefig(output / "custom_errors_before_after.png", dpi=140)
        plt.close(fig)
    (output / "examples.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"Saved {len(selected)} validation mistakes and {len(mistakes)} custom mistakes to {output}")


if __name__ == "__main__":
    main()

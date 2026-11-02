"""CLI: python -m digit_classifier {train,evaluate,collect,predict}."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import CustomDigits, mnist_data, write_json
from .engine import device_for, file_hash, load_checkpoint, seed_everything, train
from .images import drawing_tensor, prepare_drawing
from .metrics import evaluate
from .reports import evaluation_artifacts


def parser():
    cli = argparse.ArgumentParser(description="Лабораторная №3: CNN для цифр 0–9")
    commands = cli.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Обучение на MNIST train/validation")
    fit.add_argument("--epochs", type=int, default=6)
    fit.add_argument("--batch-size", type=int, default=128)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    fit.add_argument("--data-dir", type=Path, default=Path("data"))
    fit.add_argument("--split", type=Path, default=Path("splits/mnist_seed42.json"))
    fit.add_argument("--output", type=Path, default=Path("results/baseline"))
    fit.add_argument("--download", action="store_true", help="Разрешить скачивание MNIST")
    score = commands.add_parser("evaluate", help="Оценить зафиксированный checkpoint")
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--dataset", choices=["validation", "custom"], required=True)
    score.add_argument("--custom-dir", type=Path, default=Path("custom_test"))
    score.add_argument("--data-dir", type=Path, default=Path("data"))
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=128)
    score.add_argument("--threads", type=int, default=4)
    score.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    collect = commands.add_parser("collect", help="Собрать собственные рисунки мышью")
    collect.add_argument("--output", type=Path, default=Path("custom_test"))
    predict = commands.add_parser("predict", help="Распознать одно внешнее изображение")
    predict.add_argument("image", type=Path)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict.add_argument("--save-preprocessed", type=Path)
    return cli


def score(args):
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("Batch size и threads должны быть положительными.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Папка отчёта не пуста. Укажите новую --output.")
    device = device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    seed_everything(checkpoint["config"]["seed"], args.threads)
    if args.dataset == "validation":
        split_path = args.checkpoint.parent / "split.json"
        if file_hash(split_path) != checkpoint["split_sha256"]:
            raise ValueError("Разбиение не соответствует checkpoint.")
        _, dataset, _ = mnist_data(args.data_dir, split_path, checkpoint["config"]["seed"])
        name = "MNIST validation"
        provenance = {"split_sha256": checkpoint["split_sha256"]}
    else:
        dataset = CustomDigits(args.custom_dir)
        name = "Собственный независимый тест"
        provenance = {"counts": dataset.counts,
                      "manifest_sha256": file_hash(args.custom_dir / "labels.csv"),
                      "images": [{"path": str(path.relative_to(args.custom_dir.resolve())),
                                  "label": label, "sha256": file_hash(path)}
                                 for path, label in dataset.records]}
    report, errors, predictions = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
    evaluation_artifacts(report, errors, predictions, args.output, name)
    write_json(args.output / "evaluation_config.json", {
        "checkpoint_sha256": file_hash(args.checkpoint), "epoch": checkpoint["epoch"],
        "dataset": args.dataset, "device": str(device), "batch_size": args.batch_size,
        **provenance,
    })
    print(json.dumps({k: report[k] for k in ("samples", "loss", "accuracy", "macro_f1")},
                     ensure_ascii=False, indent=2))


def predict(args):
    device = device_for(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor = drawing_tensor(image).unsqueeze(0).to(device)
        if args.save_preprocessed:
            args.save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
            prepare_drawing(image).save(args.save_preprocessed)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0].cpu()
    print(json.dumps({"digit": int(probabilities.argmax()),
                      "probabilities": {str(i): float(p) for i, p in enumerate(probabilities)}}, indent=2))


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.command == "train":
            train(args)
        elif args.command == "evaluate":
            score(args)
        elif args.command == "predict":
            predict(args)
        else:
            from .collect import launch
            import tkinter as tk
            try:
                launch(args.output)
            except tk.TclError as error:
                raise RuntimeError(
                    "Не удалось запустить Tkinter. Запустите команду в обычном PowerShell "
                    "вне песочницы; если ошибка сохраняется, проверьте установку Tcl/Tk "
                    f"в Python. Подробности: {error}"
                ) from error
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        cli.exit(2, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()

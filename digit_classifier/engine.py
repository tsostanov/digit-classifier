"""Обучение и переносимое сохранение state_dict."""

import csv
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .data import mnist_data, write_json
from .images import PREPROCESSING
from .metrics import evaluate
from .model import ARCHITECTURE, DigitCNN
from .reports import evaluation_artifacts, history_plots, overfitting_report


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна: установите CUDA-сборку PyTorch или выберите --device cpu.")
    return torch.device(name)


def seed_everything(seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, model, epoch, validation_loss, config, split_sha256):
    torch.save({
        "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": epoch, "validation_loss": validation_loss,
        "config": config, "split_sha256": split_sha256,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["architecture"] != ARCHITECTURE or checkpoint["preprocessing"] != PREPROCESSING:
        raise ValueError("Несовместимая архитектура или предобработка checkpoint.")
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def train(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Папка {output} не пуста. Выберите новую --output для сохранения эксперимента.")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        raise ValueError("Эпохи, batch size, learning rate и threads должны быть положительными.")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, args.threads)
    device = device_for(args.device)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "actual_device": str(device), "python": platform.python_version(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "numpy": np.__version__, "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
        "optimizer": "Adam", "loss": "CrossEntropyLoss", "dropout": 0.25,
    })
    write_json(output / "config.json", config)
    train_data, val_data, split = mnist_data(args.data_dir, args.split, args.seed, args.download)
    write_json(output / "split.json", split)
    split_hash = file_hash(output / "split.json")
    counts = {name: torch.bincount(dataset.tensors[1], minlength=10).tolist()
              for name, dataset in (("train", train_data), ("validation", val_data))}
    write_json(output / "split_summary.json", {"counts": counts, "split_sha256": split_hash,
                                               "overlap": 0, "total": 60000})
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    train_eval = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    model = DigitCNN().to(device)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    write_json(output / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_epoch, history = float("inf"), 0, []
    print(f"Устройство: {device}; параметров: {config['parameters']}; train=50000, validation=10000", flush=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimization_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            optimization_loss += loss.item() * len(labels)
        train_report, _, _ = evaluate(model, train_eval, device, max_errors=0)
        val_report, _, _ = evaluate(model, val_loader, device, max_errors=0)
        row = {"epoch": epoch, "optimization_loss": optimization_loss / len(train_data),
               "seconds": time.perf_counter() - epoch_started}
        for prefix, report in (("train", train_report), ("validation", val_report)):
            row.update({f"{prefix}_{metric}": report[metric] for metric in ("loss", "accuracy", "macro_f1")})
        history.append(row)
        if val_report["loss"] < best_loss:
            best_loss, best_epoch = val_report["loss"], epoch
            save_checkpoint(output / "best.pt", model, epoch, best_loss, config, split_hash)
        write_json(output / "history.json", history)
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        history_plots(history, output)
        print(f"Эпоха {epoch}/{args.epochs}: train loss={train_report['loss']:.5f}, "
              f"val loss={val_report['loss']:.5f}, val accuracy={val_report['accuracy']:.4%}, "
              f"val F1={val_report['macro_f1']:.5f}, {row['seconds']:.1f} с", flush=True)
    model, checkpoint = load_checkpoint(output / "best.pt", device)
    report, errors, predictions = evaluate(model, val_loader, device)
    evaluation_artifacts(report, errors, predictions, output / "validation", "MNIST validation")
    overfitting_report(history, best_epoch, output)
    write_json(output / "summary.json", {
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "validation_accuracy": report["accuracy"], "validation_macro_f1": report["macro_f1"],
        "seconds": time.perf_counter() - started, "checkpoint_sha256": file_hash(output / "best.pt"),
        "split_sha256": split_hash, "custom_test_status": "not_collected_not_evaluated",
        "checkpoint_reload_loss_matches": abs(report["loss"] - checkpoint["validation_loss"]) < 1e-7,
    })
    print(f"Готово. Лучшая эпоха: {best_epoch}. Отчёты: {output}", flush=True)

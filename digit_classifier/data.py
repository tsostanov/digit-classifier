"""Воспроизводимое разбиение MNIST и строгая проверка собственного теста."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST

from .images import MEAN, STD, drawing_tensor


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def labels_hash(labels):
    return hashlib.sha256(np.asarray(labels, dtype=np.uint8).tobytes()).hexdigest()


def validate_split(split, labels, seed):
    labels = np.asarray(labels)
    if split["seed"] != seed or split["labels_sha256"] != labels_hash(labels):
        raise ValueError("Seed или контрольная сумма меток не совпадают с разбиением.")
    train, val = np.asarray(split["train"]), np.asarray(split["validation"])
    if len(train) != 50000 or len(val) != 10000:
        raise ValueError("Неверные размеры разбиения.")
    joined = np.concatenate((train, val))
    if not np.issubdtype(joined.dtype, np.integer) or not np.array_equal(
        np.sort(joined), np.arange(len(labels))
    ):
        raise ValueError("Индексы повторяются, пересекаются или не покрывают MNIST.")
    counts = np.bincount(labels, minlength=10)
    actual = np.bincount(labels[val], minlength=10)
    if np.any(np.abs(actual - counts / 6) >= 1):
        raise ValueError("Нарушены пропорции классов.")


def get_split(labels, path, seed=42):
    path = Path(path)
    labels = np.asarray(labels)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
    else:
        if len(labels) != 60000 or set(labels.tolist()) != set(range(10)):
            raise ValueError("Ожидались 60 000 меток MNIST, классы 0–9.")
        rng = np.random.default_rng(seed)
        exact = np.bincount(labels, minlength=10) / 6
        quotas = np.floor(exact).astype(int)
        order = np.argsort(-(exact - quotas), kind="stable")
        quotas[order[:10000 - int(quotas.sum())]] += 1
        train, val = [], []
        for label, quota in enumerate(quotas):
            indices = rng.permutation(np.flatnonzero(labels == label))
            val.extend(indices[:quota].tolist())
            train.extend(indices[quota:].tolist())
        split = {
            "dataset": "torchvision.MNIST(train=True)", "seed": seed,
            "labels_sha256": labels_hash(labels),
            "train": rng.permutation(train).tolist(),
            "validation": rng.permutation(val).tolist(),
        }
        validate_split(split, labels, seed)
        write_json(path, split)
    validate_split(split, labels, seed)
    return split


def mnist_data(root, split_path, seed, download=False):
    mnist = MNIST(root=str(root), train=True, download=download)
    split = get_split(mnist.targets.numpy(), split_path, seed)
    # Та же операция, что ToTensor + Normalize; один раз в RAM для быстрого CPU.
    images = mnist.data.unsqueeze(1).float().div_(255).sub_(MEAN).div_(STD)
    datasets = []
    for name in ("train", "validation"):
        indices = torch.tensor(split[name], dtype=torch.long)
        datasets.append(TensorDataset(images[indices], mnist.targets[indices]))
    return *datasets, split


def read_custom(root, require_complete=True):
    root = Path(root).resolve()
    manifest = root / "labels.csv"
    if not manifest.is_file():
        raise ValueError(f"Нет {manifest}. Сначала соберите рисунки командой collect.")
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV должен содержать столбцы path,label.")
        rows = list(reader)
    records, paths, hashes = [], set(), set()
    counts = [0] * 10
    for row in rows:
        if row["label"] not in [str(i) for i in range(10)]:
            raise ValueError(f"Некорректная метка: {row['label']!r}.")
        if not row["path"]:
            raise ValueError("В CSV указан пустой путь изображения.")
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path in paths:
            raise ValueError(f"Недопустимый или повторный путь: {relative}.")
        if path.suffix.lower() != ".png" or not path.is_file():
            raise ValueError(f"Не найден PNG: {relative}.")
        with Image.open(path) as image:
            digest = hashlib.sha256(image.convert("L").tobytes() + str(image.size).encode()).hexdigest()
            drawing_tensor(image)  # заранее проверить читаемость и непустой рисунок
        if digest in hashes:
            raise ValueError(f"Дубликат рисунка: {relative}.")
        hashes.add(digest)
        paths.add(path)
        label = int(row["label"])
        counts[label] += 1
        records.append((path, label))
    unlabelled = {p.resolve() for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() == ".png"} - paths
    if unlabelled:
        raise ValueError(f"Есть PNG без разметки: {len(unlabelled)}.")
    if require_complete and (len(records) < 50 or min(counts) < 5):
        raise ValueError(f"Нужно минимум 5 рисунков каждого класса (всего ≥50). Сейчас: {counts}.")
    return records, counts


class CustomDigits(Dataset):
    def __init__(self, root):
        self.records, self.counts = read_custom(root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            tensor = drawing_tensor(image)
        return tensor, label

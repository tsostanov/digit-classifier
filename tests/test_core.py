"""Синтетические фикстуры проверяют код и никогда не попадают в custom_test."""

import csv
import json
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, TensorDataset

from digit_classifier.collect import save_drawing
from digit_classifier.data import CustomDigits, get_split, read_custom, validate_split
from digit_classifier.engine import load_checkpoint, save_checkpoint, seed_everything, train
from digit_classifier.images import MEAN, STD, drawing_tensor, prepare_drawing
from digit_classifier.metrics import classification_metrics, evaluate
from digit_classifier.model import DigitCNN


@pytest.fixture(autouse=True)
def deterministic():
    seed_everything(42, threads=2)


def stroke(offset=0):
    image = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(image)
    draw.line([(80 + offset, 50), (150, 130), (110, 230)], fill=255, width=20)
    return image


def test_split_is_stratified_complete_and_persistent(tmp_path):
    # Реальные размеры классов MNIST, но здесь только искусственный вектор меток.
    counts = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    labels = np.repeat(np.arange(10), counts)
    path = tmp_path / "split.json"
    split = get_split(labels, path, 42)
    assert len(split["train"]) == 50000
    assert len(split["validation"]) == 10000
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == set(range(60000))
    np.testing.assert_array_less(
        np.abs(np.bincount(labels[split["validation"]]) - np.array(counts) / 6), 1
    )
    assert get_split(labels, path, 42) == split
    assert get_split(labels, tmp_path / "same.json", 42) == split
    with pytest.raises(ValueError, match="Seed"):
        get_split(labels, path, 1)
    split["validation"][0] = split["train"][0]
    with pytest.raises(ValueError, match="пересекаются"):
        validate_split(split, labels, 42)


def test_model_output_and_optimizer_step():
    model = DigitCNN()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 8, 9])
    logits = model(images)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.layers[0].weight.detach().clone()
    torch.nn.CrossEntropyLoss()(logits, labels).backward()
    optimizer.step()
    assert not torch.equal(before, model.layers[0].weight)


def test_checkpoint_roundtrip(tmp_path):
    model = DigitCNN().eval()
    images = torch.randn(3, 1, 28, 28)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, 2, 0.3, {"seed": 42}, "abc")
    restored, checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert not restored.training
    assert checkpoint["epoch"] == 2
    torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)


def test_drawing_preprocessing_polarity_shape_and_center():
    image = stroke()
    prepared = prepare_drawing(image)
    tensor = drawing_tensor(image)
    assert prepared.size == (28, 28)
    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) == pytest.approx(-MEAN / STD)
    torch.testing.assert_close(tensor, drawing_tensor(ImageOps.invert(image)))
    pixels = np.asarray(prepared)
    yy, xx = np.indices(pixels.shape)
    assert abs((pixels * xx).sum() / pixels.sum() - 13.5) <= 0.6
    assert abs((pixels * yy).sum() / pixels.sum() - 13.5) <= 0.6


@pytest.mark.parametrize("color", [0, 255])
def test_blank_rejected_before_save(tmp_path, color):
    with pytest.raises(ValueError, match="Пустой"):
        save_drawing(Image.new("L", (280, 280), color), 0, tmp_path / "test")
    assert not (tmp_path / "test").exists()


def test_transparent_image():
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(image).line((40, 10, 60, 90), fill=(0, 0, 0, 255), width=8)
    assert drawing_tensor(image).shape == (1, 28, 28)


def test_collector_manifest_duplicates_and_incomplete_test(tmp_path):
    path = save_drawing(stroke(), 7, tmp_path)
    assert path.is_file()
    records, counts = read_custom(tmp_path, require_complete=False)
    assert len(records) == 1 and counts[7] == 1
    with pytest.raises(ValueError, match="уже сохранён"):
        save_drawing(stroke(), 7, tmp_path)
    with pytest.raises(ValueError, match="5 рисунков"):
        CustomDigits(tmp_path)
    with pytest.raises(ValueError, match="Метка"):
        save_drawing(stroke(10), 10, tmp_path)


def test_custom_rejects_bad_labels_and_traversal(tmp_path):
    manifest = tmp_path / "labels.csv"
    for relative, label, message in [("x.png", "10", "метка"), ("../x.png", "1", "путь")]:
        manifest.write_text(f"path,label\n{relative},{label}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_custom(tmp_path, require_complete=False)
    manifest.write_text("path,label\n,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пустой путь"):
        read_custom(tmp_path, require_complete=False)


def test_complete_custom_dataset_only_in_temporary_directory(tmp_path):
    # Пятьдесят искусственных фигур — только для проверки загрузчика, не тест модели.
    for index in range(50):
        save_drawing(stroke(index), index // 5, tmp_path)
    dataset = CustomDigits(tmp_path)
    assert len(dataset) == 50 and dataset.counts == [5] * 10
    image, label = dataset[0]
    assert image.shape == (1, 28, 28) and label == 0
    with (tmp_path / "labels.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow([dataset.records[0][0].relative_to(tmp_path), 0, ""])
    with pytest.raises(ValueError, match="повторный"):
        read_custom(tmp_path)


def test_unlabelled_png_and_duplicate_content_rejected(tmp_path):
    path = save_drawing(stroke(), 2, tmp_path)
    unlabelled = tmp_path / "orphan.PNG"
    stroke(10).save(unlabelled)
    with pytest.raises(ValueError, match="без разметки"):
        read_custom(tmp_path, require_complete=False)
    unlabelled.unlink()
    copied = tmp_path / "copy.png"
    copied.write_bytes(path.read_bytes())
    with (tmp_path / "labels.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["copy.png", 3, ""])
    with pytest.raises(ValueError, match="Дубликат"):
        read_custom(tmp_path, require_complete=False)


def test_predict_cli_function(tmp_path, capsys):
    from digit_classifier.__main__ import predict

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, DigitCNN(), 1, 1.0, {"seed": 42}, "fixture")
    source, processed = tmp_path / "drawing.png", tmp_path / "processed.png"
    stroke().save(source)
    predict(Namespace(image=source, checkpoint=checkpoint, device="cpu", save_preprocessed=processed))
    result = json.loads(capsys.readouterr().out)
    assert result["digit"] in range(10)
    assert len(result["probabilities"]) == 10
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    with Image.open(processed) as image:
        assert image.size == (28, 28)


def test_metrics_use_all_ten_classes_and_weight_loss_by_samples():
    matrix = np.eye(10, dtype=np.int64) * 2
    matrix[0, 0] = 1
    matrix[0, 1] = 1
    report = classification_metrics(matrix)
    assert report["accuracy"] == 0.95
    assert report["macro_f1"] == pytest.approx((2 / 3 + 4 / 5 + 8) / 10)
    model = DigitCNN().eval()
    images, labels = torch.randn(5, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4])
    report, errors, predictions = evaluate(model, DataLoader(TensorDataset(images, labels), batch_size=3), "cpu")
    expected = torch.nn.functional.cross_entropy(model(images), labels).item()
    assert report["loss"] == pytest.approx(expected, abs=1e-6)
    assert report["samples"] == 5
    assert len(predictions) == 5
    assert len(errors) == sum(p != y for p, y in zip(predictions, labels.tolist()))


def test_training_artifacts_and_best_checkpoint(tmp_path, monkeypatch):
    import digit_classifier.engine as engine

    # Интеграционный тест цикла без скачивания MNIST; фикстура не является результатом лабораторной.
    images = torch.randn(20, 1, 28, 28)
    labels = torch.arange(20) % 10
    dataset = TensorDataset(images, labels)
    monkeypatch.setattr(engine, "mnist_data", lambda *args: (dataset, dataset, {"fixture": True}))
    output = tmp_path / "run"
    args = Namespace(output=output, epochs=2, batch_size=8, lr=0.001, threads=2,
                     seed=42, device="cpu", data_dir=tmp_path, split=tmp_path / "split.json", download=False)
    train(args)
    history = json.loads((output / "history.json").read_text())
    _, checkpoint = load_checkpoint(output / "best.pt", "cpu")
    assert checkpoint["validation_loss"] == min(row["validation_loss"] for row in history)
    assert (output / "learning_curves.png").is_file()
    assert (output / "validation/metrics.json").is_file()
    assert json.loads((output / "summary.json").read_text())["checkpoint_reload_loss_matches"]
    with pytest.raises(ValueError, match="не пуста"):
        train(args)

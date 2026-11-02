"""Небольшие графики и отчёты, пригодные для сохранения в Git."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import write_json
from .images import MEAN, STD


def history_plots(history, output):
    output = Path(output)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Потери", "Accuracy")):
        for prefix, label in (("train", "Обучение (eval)"), ("validation", "Валидация")):
            axis.plot(epochs, [r[f"{prefix}_{metric}"] for r in history], "o-", label=label)
        axis.set(xlabel="Эпоха", ylabel=title, xticks=epochs)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=140)
    plt.close(fig)


def evaluation_artifacts(report, errors, predictions, output, dataset):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {**report, "dataset": dataset,
              "errors_preview": [{k: v for k, v in error.items() if k != "image"} for error in errors]}
    write_json(output / "metrics.json", report)
    write_json(output / "predictions.json", predictions)
    matrix = np.array(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    color = ax.imshow(matrix, cmap="Blues")
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, str(value), ha="center", va="center", fontsize=7,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Предсказанная цифра",
           ylabel="Правильная цифра", title=f"Матрица ошибок: {dataset}")
    fig.colorbar(color, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, error in zip(axes.flat, errors):
        ax.imshow((error["image"].squeeze().numpy() * STD + MEAN), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"№{error['index']}: {error['true']} → {error['predicted']}", fontsize=9)
    if not errors:
        fig.suptitle("Ошибок нет")
    fig.tight_layout()
    fig.savefig(output / "misclassified.png", dpi=130)
    plt.close(fig)


def overfitting_report(history, best_epoch, output):
    best = history[best_epoch - 1]
    last = history[-1]
    rises = [r["epoch"] for before, r in zip(history, history[1:])
             if r["validation_loss"] > before["validation_loss"]
             and r["train_loss"] < before["train_loss"]]
    text = (
        "# Анализ переобучения по фактическому запуску\n\n"
        "Train и validation измерены на фиксированных весах после каждой эпохи "
        "в режиме eval; Dropout отключён в обоих измерениях.\n\n"
        f"Лучшая эпоха по validation loss: {best_epoch}. "
        f"Train loss={best['train_loss']:.6f}, validation loss={best['validation_loss']:.6f}; "
        f"train accuracy={best['train_accuracy']:.4%}, "
        f"validation accuracy={best['validation_accuracy']:.4%}. "
        f"Разрыв accuracy={(best['train_accuracy'] - best['validation_accuracy']) * 100:.3f} п.п.\n\n"
        f"В последнюю эпоху: train loss={last['train_loss']:.6f}, "
        f"validation loss={last['validation_loss']:.6f}. "
        f"Эпохи с ростом validation loss при снижении train loss: {rises or 'нет'}.\n\n"
    )
    if rises:
        text += ("Обнаружено локальное расхождение динамики потерь train и validation. "
                 "Одиночный рост может быть флуктуацией и не доказывает устойчивое "
                 "переобучение. Учитывайте величину изменений, accuracy и последующие эпохи. "
                 "Используется лучший checkpoint.\n\n")
    else:
        text += ("На измеренном промежутке нет одновременного роста validation loss и снижения "
                 "train loss. Это не гарантирует отсутствия переобучения при более долгом обучении.\n\n")
    text += ("Validation используется для выбора checkpoint и не является независимым итоговым "
             "тестом. Собственный тест ещё не проведён: пользователь должен собрать рисунки. "
             "Перенос качества с MNIST на рисунки мышью до этого не установлен.\n")
    Path(output, "overfitting.md").write_text(text, encoding="utf-8")

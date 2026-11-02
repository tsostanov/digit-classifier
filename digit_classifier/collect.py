"""Пользователь вручную рисует и размечает цифры; модель здесь не используется."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from .data import read_custom
from .images import prepare_drawing


def save_drawing(image, label, root):
    if type(label) is not int or label not in range(10):
        raise ValueError("Метка должна быть целым числом от 0 до 9.")
    prepare_drawing(image)  # отклонить пустой холст до записи на диск
    root = Path(root)
    manifest = root / "labels.csv"
    records, _ = read_custom(root, require_complete=False) if manifest.exists() else ([], [0] * 10)
    gray = image.convert("L")
    for path, _ in records:
        with Image.open(path) as existing:
            if existing.size == gray.size and existing.convert("L").tobytes() == gray.tobytes():
                raise ValueError("Этот рисунок уже сохранён. Нарисуйте новый пример.")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    relative = Path("raw") / f"{label}_{uuid4().hex}.png"
    gray.save(root / relative)
    try:
        header = not manifest.exists()
        with manifest.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "created_at_utc"])
            if header:
                writer.writeheader()
            writer.writerow({"path": relative.as_posix(), "label": label,
                             "created_at_utc": datetime.now(timezone.utc).isoformat()})
    except Exception:
        (root / relative).unlink()
        raise
    return root / relative


class Collector:
    SIZE = 280
    WIDTH = 20

    def __init__(self, window, output):
        import tkinter as tk
        from tkinter import ttk

        self.window, self.output = window, Path(output)
        window.title("Лабораторная №3 — собственные рукописные цифры")
        window.resizable(False, False)
        ttk.Label(window, text="Нарисуйте одну цифру мышью и укажите правильную метку.",
                  padding=10).pack()
        self.canvas = tk.Canvas(window, width=self.SIZE, height=self.SIZE, bg="black",
                                highlightthickness=1)
        self.canvas.pack(padx=15)
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.stroke)
        self.canvas.bind("<ButtonRelease-1>", lambda event: setattr(self, "previous", None))
        controls = ttk.Frame(window, padding=10)
        controls.pack()
        ttk.Label(controls, text="Правильная цифра:").grid(row=0, column=0, padx=5)
        self.label = tk.StringVar(value="0")
        ttk.Combobox(controls, textvariable=self.label, values=list(range(10)),
                     state="readonly", width=3).grid(row=0, column=1)
        ttk.Button(controls, text="Сохранить", command=self.save).grid(row=0, column=2, padx=8)
        ttk.Button(controls, text="Очистить", command=self.clear).grid(row=0, column=3)
        self.count_text = tk.StringVar()
        ttk.Label(window, textvariable=self.count_text, padding=10, justify="left").pack()
        self.status = tk.StringVar(value=f"Папка: {self.output.resolve()}")
        ttk.Label(window, textvariable=self.status, padding=10, wraplength=490).pack()
        ttk.Label(window, text="Цель: ≥5 разных рисунков каждой цифры, всего ≥50.\n"
                  "Собирайте примеры до просмотра предсказаний модели.", padding=8).pack()
        self.clear()
        self.update_counts()

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.SIZE, self.SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.previous = None

    def point(self, event):
        return max(0, min(self.SIZE - 1, event.x)), max(0, min(self.SIZE - 1, event.y))

    def start(self, event):
        x, y = self.point(event)
        radius = self.WIDTH / 2
        box = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*box, fill="white", outline="white")
        self.draw.ellipse(box, fill=255)
        self.previous = (x, y)

    def stroke(self, event):
        point = self.point(event)
        if self.previous is not None:
            self.canvas.create_line(*self.previous, *point, fill="white", width=self.WIDTH,
                                    capstyle="round", smooth=True)
            self.draw.line([self.previous, point], fill=255, width=self.WIDTH)
        self.start(event)

    def update_counts(self):
        if (self.output / "labels.csv").exists():
            _, counts = read_custom(self.output, require_complete=False)
        else:
            counts = [0] * 10
        lines = ["   ".join(f"{label}: {counts[label]}/5" for label in range(start, start + 5))
                 for start in (0, 5)]
        self.count_text.set("\n".join(lines) + f"\nВсего: {sum(counts)} / 50")

    def save(self):
        from tkinter import messagebox

        try:
            path = save_drawing(self.image, int(self.label.get()), self.output)
            self.update_counts()
        except (ValueError, OSError) as error:
            messagebox.showerror("Не удалось сохранить", str(error), parent=self.window)
            return
        self.clear()
        self.status.set(f"Сохранено: {path.name}. Нарисуйте следующий пример.")


def launch(output):
    import tkinter as tk

    window = tk.Tk()
    Collector(window, output)
    window.mainloop()

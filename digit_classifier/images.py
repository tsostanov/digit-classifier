"""Подготовка рисунков к формату MNIST: 28×28, светлая цифра на тёмном фоне."""

import numpy as np
import torch
from PIL import Image, ImageOps

MEAN = 0.1307
STD = 0.3081
PREPROCESSING = "mnist-normalize_v1_custom-crop20-center-mass"


def prepare_drawing(image: Image.Image) -> Image.Image:
    """Обрезать фон, уменьшить цифру и разместить её в центре поля 28×28."""
    # Прозрачные изображения сначала размещаем на белом фоне.
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(bg, rgba)
    gray = image.convert("L")
    pixels = np.asarray(gray)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) > 127:
        gray = ImageOps.invert(gray)
    pixels = np.array(gray, copy=True)
    pixels[pixels < 30] = 0  # удаление слабого фонового шума
    if np.count_nonzero(pixels) < 10 or pixels.max() < 50:
        raise ValueError("Пустой или слишком маленький рисунок: нарисуйте цифру.")
    gray = Image.fromarray(pixels)
    cropped = gray.crop(gray.getbbox())
    scale = 20 / max(cropped.size)
    resized = cropped.resize(
        tuple(max(1, round(v * scale)) for v in cropped.size), Image.Resampling.LANCZOS
    )
    result = Image.new("L", (28, 28), 0)
    result.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    # Совмещаем центр светлых штрихов с серединой картинки.
    pixels = np.asarray(result, dtype=np.float64)
    rows, columns = np.indices(pixels.shape)
    total_brightness = pixels.sum()
    center_x = float((pixels * columns).sum() / total_brightness)
    center_y = float((pixels * rows).sum() / total_brightness)
    shift_x = round(13.5 - center_x)
    shift_y = round(13.5 - center_y)
    centered = Image.new("L", (28, 28), 0)
    centered.paste(result, (shift_x, shift_y))
    return centered


def drawing_tensor(image: Image.Image) -> torch.Tensor:
    prepared = prepare_drawing(image)
    values = torch.from_numpy(np.array(prepared, dtype=np.float32)).unsqueeze(0) / 255
    return (values - MEAN) / STD

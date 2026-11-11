from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WORK = 56

_SEGMENTS = {
    0: [(14, 8, 40, 8), (40, 8, 40, 46), (40, 46, 14, 46), (14, 46, 14, 8)],
    1: [(28, 8, 28, 46)],
    2: [(14, 10, 40, 10), (40, 10, 40, 26), (40, 26, 14, 26),
        (14, 26, 14, 44), (14, 44, 40, 44)],
    3: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44), (20, 27, 40, 27)],
    4: [(14, 10, 14, 30), (14, 30, 40, 30), (40, 10, 40, 46)],
    5: [(40, 10, 14, 10), (14, 10, 14, 27), (14, 27, 40, 27),
        (40, 27, 40, 44), (40, 44, 14, 44)],
    6: [(40, 10, 16, 10), (16, 10, 16, 44), (16, 44, 40, 44),
        (40, 44, 40, 27), (40, 27, 16, 27)],
    7: [(14, 10, 40, 10), (40, 10, 20, 46)],
    8: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44),
        (14, 44, 14, 10), (14, 27, 40, 27)],
    9: [(40, 44, 40, 12), (40, 12, 16, 12), (16, 12, 16, 27),
        (16, 27, 40, 27), (16, 12, 16, 10)],
}


def _draw(digit: int, rng: np.random.Generator) -> Image.Image:
    layer = Image.new("L", (WORK, WORK), 0)
    pen = ImageDraw.Draw(layer)
    width = int(rng.integers(2, 5))
    for x1, y1, x2, y2 in _SEGMENTS[digit]:
        pen.line((x1, y1, x2, y2), fill=255, width=width)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.0))))
    layer = layer.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR, fillcolor=0)
    dy, dx = (int(v) for v in rng.integers(-3, 4, size=2))
    return Image.fromarray(np.roll(np.array(layer), (dy, dx), axis=(0, 1)))


def _fit(layer: Image.Image, side: int, rng: np.random.Generator) -> np.ndarray:
    raw = np.asarray(layer, dtype=np.float32) / 255.0
    rows, cols = np.nonzero(raw > 0.1)
    if rows.size == 0:
        return np.zeros((side, side), dtype=np.float32)

    boxed = raw[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    image = Image.fromarray((boxed * 255).astype(np.uint8))

    target = int(side * 0.72)
    w, h = image.size
    scale = target / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    board = Image.new("L", (side, side), 0)
    board.paste(image, ((side - new_w) // 2, (side - new_h) // 2))
    base = np.asarray(board, dtype=np.float32) / 255.0
    noisy = base + rng.normal(0, 0.03, base.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def make_dataset(per_digit: int, side: int = 28, seed: int = 42):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for digit in range(10):
        for _ in range(per_digit):
            xs.append(_fit(_draw(digit, rng), side, rng))
            ys.append(digit)
    X = np.stack(xs).astype(np.float32)[:, None, :, :]
    y = np.asarray(ys, dtype=np.int64)
    return X, y
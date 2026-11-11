from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch
from PIL import Image

from engine import load_model, train
from settings import Settings, load_settings
from synth import make_dataset


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _to_tensor(image: Image.Image, side: int) -> torch.Tensor:
    gray = image.convert("L").resize((side, side), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr)[None, None, :, :]


def _probs(net, batch, device):
    with torch.no_grad():
        return torch.softmax(net(batch.to(device)), dim=1).cpu().numpy()


def _run_demo(cfg: Settings, net, device) -> None:
    X, y = make_dataset(5, cfg.image_side, seed=7)
    probs = _probs(net, torch.from_numpy(X), device)
    preds = probs.argmax(1)
    hits = int((preds == y).sum())
    print(f"demo: {hits}/{len(y)} correct")
    for truth, pred, conf in zip(y.tolist(), preds.tolist(), probs.max(1)):
        flag = "ok" if truth == pred else "!!"
        print(f"  {flag} true={truth} pred={pred} p={conf:.3f}")


def _run_files(cfg: Settings, net, device, paths: list[str]) -> None:
    for path in paths:
        tensor = _to_tensor(Image.open(path), cfg.image_side)
        probs = _probs(net, tensor, device)[0]
        print(f"{path}: {int(probs.argmax())}  (p={probs.max():.3f})")


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(prog="glyphsort")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="train the network")

    pred = sub.add_parser("predict", help="recognize digits")
    pred.add_argument("images", nargs="*")
    pred.add_argument("--demo", action="store_true")

    args = parser.parse_args(argv)
    cfg = load_settings(args.config)

    if args.cmd == "train":
        train(cfg)
        return 0

    if args.cmd == "predict":
        if not args.demo and not args.images:
            parser.error("predict requires --demo or image paths")
        net, device, meta = load_model(cfg)
        logging.getLogger("glyphsort").info(
            "loaded checkpoint: epoch=%s best_val=%.2f%%",
            meta.get("epoch"), meta.get("best_val", 0) * 100,
        )
        if args.demo:
            _run_demo(cfg, net, device)
        else:
            _run_files(cfg, net, device, args.images)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
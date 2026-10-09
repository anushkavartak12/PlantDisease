"""Fine-tune MobileNetV3 on data/images and save runs/best.pt.

    python train.py --epochs 12
"""

import argparse
import os

# Each dataloader worker would otherwise start a BLAS thread pool and run out of memory on small machines.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageFilter
from scipy import ndimage
from torchvision import datasets, transforms as T

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "images"
RUNS = ROOT / "runs"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ARCH = "mobilenetv3_large_100"

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.45, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.35, 0.35, 0.3, 0.03),
    T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
# Matches the app: centre square crop, resized to 224.
eval_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def is_field(path: str) -> bool:
    return os.path.basename(path).startswith("field_")


NATURAL_TONES = [(90, 70, 50), (110, 120, 60), (60, 90, 50), (150, 150, 140), (130, 170, 210), (40, 50, 30),
                 (170, 140, 100), (80, 100, 40), (120, 100, 80)]


def synthetic_background(size: int) -> Image.Image:
    """Soil/grass/sky-toned background with a gradient and soft noise, for when no photo background is picked."""
    base = np.array(random.choice(NATURAL_TONES), dtype=np.float32) + np.random.uniform(-25, 25, 3)
    grad = np.linspace(-1, 1, size, dtype=np.float32)[:, None, None] * random.uniform(0, 50)
    noise = np.random.normal(0, random.uniform(5, 30), (size // 6, size // 6)).astype(np.float32)
    noise = np.array(Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8)).resize((size, size), Image.BICUBIC))
    arr = base + grad + (noise[..., None].astype(np.float32) - 128)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def swap_background(im: Image.Image, bg_pool: list[str]) -> Image.Image:
    """If `im` is a leaf on a plain backdrop (PlantVillage grey, white sheets), replace the backdrop.

    The backdrop is the border-connected region close to the median border colour. Photos whose border is not
    mostly one colour are left alone.
    """
    arr = np.asarray(im, dtype=np.float32)
    h, w, _ = arr.shape
    b = max(2, min(h, w) // 40)
    border = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                             arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    median = np.median(border, 0)
    # Plain backdrops can carry shadows, gradients or a leaf touching the edge, so only require that most of the
    # border is close to its median colour. Busy borders (grass, soil) fail this and are left alone.
    if (np.abs(border - median).max(1) < 40).mean() < 0.6:
        return im
    dist = np.abs(arr - median).max(2)
    lab, _ = ndimage.label(dist < random.uniform(28, 48))
    edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge_ids[edge_ids > 0])
    if bg.mean() < 0.05 or bg.mean() > 0.85:
        return im
    mask = Image.fromarray((ndimage.binary_opening(bg, iterations=1) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    if bg_pool and random.random() < 0.75:
        src = Image.open(random.choice(bg_pool)).convert("RGB")
        sw, sh = src.size
        side = random.randint(min(sw, sh) // 3, min(sw, sh))
        x, y = random.randint(0, sw - side), random.randint(0, sh - side)
        new_bg = src.crop((x, y, x + side, y + side)).resize((w, h), Image.BICUBIC)
        if random.random() < 0.5:
            new_bg = new_bg.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        new_bg = synthetic_background(max(h, w)).crop((0, 0, w, h))
    return Image.composite(new_bg, im, mask)


class LeafFolder(datasets.ImageFolder):
    """ImageFolder that can swap plain backgrounds of lab photos for real ones while training."""

    def __init__(self, root, transform, swap_p: float = 0.0, bg_pool: list[str] | None = None):
        super().__init__(root, transform)
        self.swap_p, self.bg_pool = swap_p, bg_pool or []

    def __getitem__(self, i):
        path, y = self.samples[i]
        im = self.loader(path)
        if self.swap_p and not is_field(path) and random.random() < self.swap_p:
            im = swap_background(im, self.bg_pool)
        return self.transform(im), y


def split(ds: datasets.ImageFolder, val_frac: float = 0.15, seed: int = 0):
    by_class = defaultdict(list)
    for i, (_, y) in enumerate(ds.samples):
        by_class[y].append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for idx in by_class.values():
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_frac))
        val_idx += idx[:k]
        train_idx += idx[k:]
    return train_idx, val_idx


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def evaluate(model, loader, classes, device):
    """Returns overall accuracy, per-class accuracy, per-sample class hits and per-sample crop hits.

    A crop hit is the leaf-type question on its own: summing the class probabilities per crop (as the app does)
    and checking that the most likely crop is right, whichever disease was predicted.
    """
    n = len(classes)
    crops = sorted({crop_of(c) for c in classes})
    to_crop = torch.zeros(n, len(crops))
    for k, c in enumerate(classes):
        to_crop[k, crops.index(crop_of(c))] = 1
    true_crop = torch.tensor([crops.index(crop_of(c)) for c in classes])
    model.eval()
    correct, total = torch.zeros(n), torch.zeros(n)
    hits, crop_hits = [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        for x, y in loader:
            probs = model(x.to(device, non_blocking=True)).float().softmax(1).cpu()
            pred, crop_pred = probs.argmax(1), (probs @ to_crop).argmax(1)
            for t, p, cp in zip(y, pred, crop_pred):
                total[t] += 1
                correct[t] += int(t == p)
                hits.append(int(t == p))
                crop_hits.append(int(true_crop[t] == cp))
    return (correct.sum() / total.sum()).item(), (correct / total.clamp(min=1)).tolist(), hits, crop_hits


def seed_worker(_):
    np.random.seed(torch.initial_seed() % 2**32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--swap-p", type=float, default=0.6, help="chance a plain-background photo gets a new backdrop")
    ap.add_argument("--field-repeat", type=int, default=2, help="times field photos appear per epoch")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - args.workers))
    base = datasets.ImageFolder(DATA)
    classes = base.classes
    train_idx, val_idx = split(base)
    # Backgrounds come from training-split field photos only, so validation stays unseen.
    bg_pool = [base.samples[i][0] for i in train_idx if is_field(base.samples[i][0])]
    train_idx += [i for i in train_idx for _ in range(args.field_repeat - 1) if is_field(base.samples[i][0])]
    train_ds = Subset(LeafFolder(DATA, train_tf, args.swap_p, bg_pool), train_idx)
    val_ds = Subset(LeafFolder(DATA, eval_tf), val_idx)
    field_val = [k for k, i in enumerate(val_idx) if is_field(base.samples[i][0])]
    print(f"{len(classes)} classes, {len(train_idx)} train / {len(val_idx)} val "
          f"({len(field_val)} field), {len(bg_pool)} background photos, device={device}")

    kw = dict(worker_init_fn=seed_worker, num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch * 2)  # in-process: saves ~500 MB per idle worker

    model = timm.create_model(ARCH, pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    RUNS.mkdir(exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        acc, per_class, hits, crop_hits = evaluate(model, val_dl, classes, device)
        field_acc = sum(hits[k] for k in field_val) / max(1, len(field_val))
        crop_acc = sum(crop_hits) / len(crop_hits)
        field_crop_acc = sum(crop_hits[k] for k in field_val) / max(1, len(field_val))
        # field photos are what the app really sees; leaf type is a headline feature, so it counts too
        score = (acc + field_acc + field_crop_acc) / 3 if field_val else acc
        print(f"epoch {epoch:2d}  loss {running / len(train_dl):.3f}  val_acc {acc:.4f}  field_val_acc {field_acc:.4f}  "
              f"crop_acc {crop_acc:.4f}  field_crop_acc {field_crop_acc:.4f}  ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"arch": ARCH, "classes": classes, "state_dict": model.state_dict(), "val_idx": val_idx},
                       RUNS / "best.pt")
            metrics = {"val_acc": acc, "field_val_acc": field_acc, "crop_acc": crop_acc,
                       "field_crop_acc": field_crop_acc, "per_class": dict(zip(classes, per_class))}
            (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nbest val acc {best:.4f}")
    for c, a in json.loads((RUNS / "metrics.json").read_text())["per_class"].items():
        print(f"  {c:40s} {a:.3f}")


if __name__ == "__main__":
    main()

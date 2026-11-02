"""Train the leaf models on data/images and save runs/<task>/best.pt.

The app runs two stages, so there are six models:

    python train.py --task router --size 224              # which crop is this leaf? (5 crops + "other")
    python train.py --task tomato --size 288 --init runs/router/best.pt   # healthy or which disease, per crop
    ... same for cotton, potato, rice, soybean
    python train.py --task all                            # optional: the old single 28-class model, as a baseline

All tasks use the same train/validation split, so evaluate.py can score the whole pipeline on unseen photos.
"""

import argparse
import os

# Each dataloader worker would otherwise start a BLAS thread pool and run out of memory on small machines.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from timm.utils import ModelEmaV3
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import datasets

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "images"
MASKS = ROOT / "data" / "masks"
RUNS = ROOT / "runs"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ARCH = "mobilenetv3_large_100"
CROPS = ["cotton", "potato", "rice", "soybean", "tomato"]
OTHER = "other"


# Images are augmented with PIL, NumPy and Python's `random` only, and handed over as uint8 arrays: loading runs in
# threads, and torch calls from several threads at once crash on Windows. `to_input` normalises a batch on the GPU.


def augment(im: Image.Image, size: int) -> np.ndarray:
    """Random crop (20-100% of the photo, so close-ups of spots are seen too), flips, turn, colour and focus changes."""
    w, h = im.size
    for _ in range(10):
        area = w * h * random.uniform(0.2, 1.0)
        ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw, ch = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
        if cw <= w and ch <= h:
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            break
    else:
        cw = ch = min(w, h)
        x, y = (w - cw) // 2, (h - ch) // 2
    im = im.resize((size, size), Image.BILINEAR, box=(x, y, x + cw, y + ch))
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.rotate(random.uniform(-30, 30), Image.BILINEAR)
    for enhance, amount in ((ImageEnhance.Brightness, 0.4), (ImageEnhance.Contrast, 0.4), (ImageEnhance.Color, 0.3)):
        im = enhance(im).enhance(random.uniform(1 - amount, 1 + amount))
    if random.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
    arr = np.asarray(im)
    if random.random() < 0.5:  # slight hue shift, so lighting colour varies
        hsv = np.asarray(im.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + random.randint(-8, 8)) % 256
        arr = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))
    return arr


def centre(im: Image.Image, size: int) -> np.ndarray:
    """The app's main view: centre square of the photo, resized to `size`."""
    w, h = im.size
    s = min(w, h)
    box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
    return np.asarray(im.resize((size, size), Image.BILINEAR, box=box))


_MEAN, _STD = torch.tensor(MEAN).view(1, 3, 1, 1) * 255, torch.tensor(STD).view(1, 3, 1, 1) * 255


def to_input(batch: np.ndarray, device: str) -> torch.Tensor:
    """uint8 images [N, H, W, 3] -> normalised float tensor [N, 3, H, W] on `device`."""
    x = torch.from_numpy(batch).to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x - _MEAN.to(device)) / _STD.to(device)


def is_field(path: str) -> bool:
    return os.path.basename(path).startswith("field_")


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def mask_path(path: str) -> Path:
    p = Path(path)
    return MASKS / p.parent.name / (p.stem + ".png")


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


def background(w: int, h: int, bg_pool: list[str]) -> Image.Image:
    """A random patch of a real field photo (or a synthetic one), w x h."""
    if bg_pool and random.random() < 0.85:
        src = Image.open(random.choice(bg_pool)).convert("RGB")
        sw, sh = src.size
        side = random.randint(min(sw, sh) // 3, min(sw, sh))
        x, y = random.randint(0, sw - side), random.randint(0, sh - side)
        bg = src.crop((x, y, x + side, y + side)).resize((w, h), Image.BICUBIC)
        if random.random() < 0.5:
            bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:  # out-of-focus background, like a phone focused on the leaf
            bg = bg.filter(ImageFilter.GaussianBlur(random.uniform(1, 4)))
        return bg
    return synthetic_background(max(w, h)).crop((0, 0, w, h))


def paste_on_background(im: Image.Image, mask: Image.Image, bg_pool: list[str]) -> Image.Image:
    """Cut the leaf out of a lab photo and place it, smaller or full size and turned, on a real background."""
    w, h = im.size
    if mask.size != im.size:
        mask = mask.resize(im.size, Image.NEAREST)
    box = mask.getbbox()
    if box is None:
        return im
    leaf, m = im.crop(box), mask.crop(box)
    angle = random.uniform(-180, 180) if random.random() < 0.5 else 0
    if angle:
        leaf = leaf.rotate(angle, Image.BICUBIC, expand=True)
        m = m.rotate(angle, Image.BICUBIC, expand=True)
    # leaf takes 45-100% of the frame (on-plant photos are often zoomed out)
    lw, lh = leaf.size
    s = random.uniform(0.45, 1.0) * min(w / lw, h / lh)
    lw, lh = max(1, round(lw * s)), max(1, round(lh * s))
    leaf, m = leaf.resize((lw, lh), Image.BICUBIC), m.resize((lw, lh), Image.BILINEAR)
    m = m.filter(ImageFilter.GaussianBlur(random.uniform(0.6, 1.5)))
    out = background(w, h, bg_pool)
    out.paste(leaf, (random.randint(0, w - lw), random.randint(0, h - lh)), m)
    return out


class Leaves(Dataset):
    """(path, target) samples as uint8 arrays. While training, lab photos with a leaf mask may get a real-world
    background."""

    def __init__(self, samples, size: int, train: bool, swap_p: float = 0.0, bg_pool: list[str] | None = None):
        self.samples, self.size, self.train = samples, size, train
        self.swap_p, self.bg_pool = swap_p, bg_pool or []
        self.has_mask = [self.swap_p > 0 and mask_path(p).exists() for p, _ in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        im = Image.open(path).convert("RGB")
        if self.has_mask[i] and random.random() < self.swap_p:
            im = paste_on_background(im, Image.open(mask_path(path)).convert("L"), self.bg_pool)
        return (augment(im, self.size) if self.train else centre(im, self.size)), y


class ThreadLoader:
    """Training batches built by a thread pool. On Windows every DataLoader worker process re-imports torch
    (~650 MB of committed memory each), which a 16 GB laptop with a browser open cannot spare. Decoding JPEGs and
    PIL transforms release the GIL, so threads keep the GPU busy without the extra memory. The threads never call
    torch (see `augment`)."""

    def __init__(self, ds, batch: int, sampler, threads: int, drop_last: bool = True):
        self.ds, self.batch, self.sampler, self.threads, self.drop_last = ds, batch, sampler, threads, drop_last

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch if self.drop_last else -(-n // self.batch)

    def _load(self, idx):
        xs, ys = zip(*(self.ds[i] for i in idx))
        return np.stack(xs), np.array(ys)

    def __iter__(self):
        idx = list(self.sampler)
        batches = deque(idx[k:k + self.batch] for k in range(0, min(len(idx), len(self) * self.batch), self.batch))
        with ThreadPoolExecutor(self.threads) as ex:
            pending = deque()
            while batches or pending:
                while batches and len(pending) < 2 * self.threads:  # bounded read-ahead
                    pending.append(ex.submit(self._load, batches.popleft()))
                yield pending.popleft().result()


def split(ds: datasets.ImageFolder, val_frac: float = 0.15, seed: int = 0):
    """Per-class random split of the whole image set, identical for every task."""
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


def task_labels(task: str, all_classes: list[str]):
    """The model's classes, and a function from an image-folder class name to the model's class (None: not used)."""
    if task == "all":
        return all_classes, lambda c: c
    if task == "router":
        names = sorted({crop_of(c) for c in all_classes})
        return names, crop_of
    names = [c for c in all_classes if crop_of(c) == task]
    if not names:
        raise SystemExit(f"unknown task {task!r}")
    return names, lambda c: c if crop_of(c) == task else None


def evaluate(model, loader, n, device):
    """Overall accuracy, per-class accuracy and per-sample hits."""
    model.eval()
    correct, total = torch.zeros(n), torch.zeros(n)
    hits = []
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        for x, y in loader:
            pred = model(to_input(x, device)).float().argmax(1).cpu().numpy()
            for t, p in zip(y, pred):
                total[t] += 1
                correct[t] += int(t == p)
                hits.append(int(t == p))
    return (correct.sum() / total.sum()).item(), (correct / total.clamp(min=1)).tolist(), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="router", help="router, all, or a crop: " + ", ".join(CROPS))
    ap.add_argument("--size", type=int, default=224, help="input width and height in pixels")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--threads", type=int, default=8, help="image loading threads")
    ap.add_argument("--init", type=Path, help="start from this checkpoint's backbone (e.g. the router)")
    ap.add_argument("--swap-p", type=float, default=0.6, help="chance a lab photo's leaf is put on a real background")
    ap.add_argument("--field-repeat", type=float, default=2, help="sampling weight of field photos vs lab photos")
    ap.add_argument("--balance", type=float, default=0.5,
                    help="sampling weight of a class ~ 1 / count**balance: 1 draws every class equally often")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - args.threads))
    base = datasets.ImageFolder(DATA)
    classes, to_task = task_labels(args.task, base.classes)
    train_all, val_all = split(base)

    def pick(idx):
        out = []
        for i in idx:
            path, y = base.samples[i]
            c = to_task(base.classes[y])
            if c is not None:
                out.append((i, path, classes.index(c)))
        return out

    train_s, val_s = pick(train_all), pick(val_all)
    # Backgrounds come from training-split field photos of every crop, so validation photos stay unseen.
    bg_pool = [base.samples[i][0] for i in train_all if is_field(base.samples[i][0])]
    train_ds = Leaves([(p, y) for _, p, y in train_s], args.size, True, args.swap_p, bg_pool)
    val_ds = Leaves([(p, y) for _, p, y in val_s], args.size, False)
    field_val = [k for k, (_, p, _) in enumerate(val_s) if is_field(p)]

    # Balanced sampling: rare classes (e.g. soybean frogeye) are drawn more often, field photos more than lab.
    counts = Counter(y for _, _, y in train_s)
    weights = [(args.field_repeat if is_field(p) else 1.0) / counts[y] ** args.balance for _, p, y in train_s]
    n_draw = int(sum(args.field_repeat if is_field(p) else 1 for _, p, _ in train_s))
    sampler = WeightedRandomSampler(weights, n_draw, replacement=True)
    print(f"task {args.task}: {len(classes)} classes {classes}\n"
          f"{len(train_s)} train ({n_draw} draws/epoch, {sum(train_ds.has_mask)} with a leaf mask) / "
          f"{len(val_s)} val ({len(field_val)} field), {len(bg_pool)} background photos, {args.size}px, {device}")

    train_dl = ThreadLoader(train_ds, args.batch, sampler, args.threads)
    val_dl = ThreadLoader(val_ds, args.batch * 2, range(len(val_ds)), args.threads, drop_last=False)

    model = timm.create_model(ARCH, pretrained=True, num_classes=len(classes), drop_rate=0.2)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
        print("init from", args.init, model.load_state_dict(sd, strict=False))
    model.to(device)
    ema = ModelEmaV3(model, decay=0.998, use_warmup=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out = RUNS / args.task
    out.mkdir(parents=True, exist_ok=True)
    best, step, first = -1.0, 0, 1
    # Resume after a crash or restart: last.pt holds everything needed to carry on from the last finished epoch.
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.module.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        best, step, first = st["best"], st["step"], st["epoch"] + 1
        print(f"resuming after epoch {st['epoch']}", flush=True)
    for epoch in range(first, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = to_input(x, device), torch.from_numpy(y).to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            ema.update(model, step)
            running += loss.item()
        acc, per_class, hits = evaluate(ema.module, val_dl, len(classes), device)
        field_acc = sum(hits[k] for k in field_val) / max(1, len(field_val))
        # field photos are what the app really sees, so they count as much as everything else together
        score = (acc + field_acc) / 2 if field_val else acc
        print(f"epoch {epoch:2d}  loss {running / len(train_dl):.3f}  val_acc {acc:.4f}  "
              f"field_val_acc {field_acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save({"arch": ARCH, "task": args.task, "classes": classes, "size": args.size,
                        "state_dict": ema.module.state_dict(), "val_idx": [i for i, _, _ in val_s]}, out / "best.pt")
            metrics = {"task": args.task, "size": args.size, "epoch": epoch, "val_acc": acc,
                       "field_val_acc": field_acc, "val_images": len(val_s), "field_val_images": len(field_val),
                       "per_class": dict(zip(classes, per_class))}
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best, "step": step,
                    "epoch": epoch}, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", out / "last.pt")  # never leave a half-written checkpoint

    (out / "last.pt").unlink()
    (out / "done").touch()
    m = json.loads((out / "metrics.json").read_text())
    print(f"\nbest epoch {m['epoch']}: val acc {m['val_acc']:.4f}, field {m['field_val_acc']:.4f}")
    for c, a in m["per_class"].items():
        print(f"  {c:40s} {a:.3f}")


if __name__ == "__main__":
    main()

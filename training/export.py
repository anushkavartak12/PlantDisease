"""Export runs/best.pt to the app:

- app/public/models/model.onnx   outputs: logits [1,C], embedding [1,D] (L2-normalised)
- app/public/models/labels.json
- app/public/models/refs.json    reference images + int8 embeddings for nearest-match lookup
- app/public/refs/*.jpg          reference thumbnails
"""

import base64
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from train import DATA, RUNS, eval_tf

ROOT = Path(__file__).parent
APP = ROOT.parent / "app" / "public"
MODELS = APP / "models"
REFS = APP / "refs"
REFS_PER_CLASS = 6
THUMB = 160


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
        logits = self.model.get_classifier()(feats)
        return logits, F.normalize(feats, dim=1)


def thumb(path: str, dest: Path):
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    w, h = im.size
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((THUMB, THUMB), Image.BICUBIC)
    im.save(dest, "JPEG", quality=80)


def main():
    ckpt = torch.load(RUNS / "best.pt", map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    wrapped = Exported(model).eval()

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, torch.randn(1, 3, 224, 224), MODELS / "model.onnx",
        input_names=["input"], output_names=["logits", "embedding"],
        opset_version=17, dynamo=False,
    )
    (MODELS / "labels.json").write_text(json.dumps(classes))
    print("exported", MODELS / "model.onnx")

    # Reference set: the most confidently-correct validation images per class.
    ds = datasets.ImageFolder(DATA, eval_tf)
    scored = defaultdict(list)
    with torch.no_grad():
        idx = ckpt["val_idx"]
        for start in range(0, len(idx), 128):
            chunk = idx[start:start + 128]
            x = torch.stack([ds[i][0] for i in chunk])
            logits, emb = wrapped(x)
            prob = logits.softmax(1)
            for j, i in enumerate(chunk):
                y = ds.samples[i][1]
                # images from datasets with no stated licence are used to train but never shipped as thumbnails
                if prob[j].argmax().item() == y and "nolic" not in Path(ds.samples[i][0]).name:
                    scored[y].append((prob[j, y].item(), i, emb[j].numpy()))

    shutil.rmtree(REFS, ignore_errors=True)
    REFS.mkdir(parents=True)
    refs = []
    for y, items in sorted(scored.items()):
        items.sort(key=lambda t: -t[0])
        # spread picks across the confident half rather than the very top few near-duplicates
        pool = items[: max(REFS_PER_CLASS, len(items) // 2)]
        picks = [pool[k * len(pool) // REFS_PER_CLASS] for k in range(min(REFS_PER_CLASS, len(pool)))]
        for k, (_, i, e) in enumerate(picks):
            name = f"{classes[y]}_{k}.jpg"
            thumb(ds.samples[i][0], REFS / name)
            q = np.clip(np.round(e * 127 / max(1e-6, np.abs(e).max())), -127, 127).astype(np.int8)
            refs.append({"id": classes[y], "img": f"refs/{name}", "emb": base64.b64encode(q.tobytes()).decode()})
    (MODELS / "refs.json").write_text(json.dumps(refs))
    print(f"wrote {len(refs)} reference images")


if __name__ == "__main__":
    main()

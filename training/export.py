"""Export the router and the five crop specialists (runs/<task>/best.pt) to the app:

- app/public/models/<task>.onnx  outputs: logits [1,C], embedding [1,D] (L2-normalised); weights stored as float16
- app/public/models/manifest.json  file, input size and labels of each model
- app/public/models/refs.json    reference photos per crop + int8 embeddings from that crop's specialist
- app/public/models/metrics.json validation scores (from evaluate.py and each model)
- app/public/refs/*.jpg          reference thumbnails
"""

import base64
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, numpy_helper
from torchvision import datasets

from train import CROPS, DATA, RUNS, centre, to_input

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


def half_weights(path: Path):
    """Store large float32 weights as float16 followed by a Cast back to float32. Halves the download; ONNX Runtime
    folds the casts when it loads the model, so inference still runs in float32."""
    m = onnx.load(path)
    g = m.graph
    casts = []
    for init in list(g.initializer):
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 256:
            g.initializer.remove(init)
            g.initializer.append(numpy_helper.from_array(arr.astype(np.float16), init.name + "_f16"))
            casts.append(helper.make_node("Cast", [init.name + "_f16"], [init.name], to=onnx.TensorProto.FLOAT))
    for node in reversed(casts):
        g.node.insert(0, node)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def thumb(path: str, dest: Path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    w, h = im.size
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((THUMB, THUMB), Image.BICUBIC)
    im.save(dest, "JPEG", quality=80)


def export(task: str):
    ckpt = torch.load(RUNS / task / "best.pt", map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    wrapped = Exported(model).eval()
    size = ckpt["size"]
    dest = MODELS / f"{task}.onnx"
    torch.onnx.export(wrapped, torch.randn(1, 3, size, size), dest, input_names=["input"],
                      output_names=["logits", "embedding"], opset_version=17, dynamo=False,
                      # the app sends several views of one photo as a batch
                      dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}, "embedding": {0: "n"}})
    half_weights(dest)

    # the float16 file must still give the PyTorch answers
    x = torch.randn(4, 3, size, size)
    sess = ort.InferenceSession(dest, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x.numpy()})[0]
    want = wrapped(x)[0].detach().numpy()
    assert (got.argmax(1) == want.argmax(1)).all() and np.abs(got - want).max() < 0.05, f"{task}: ONNX output differs"
    print(f"exported {dest.name}: {size}px, {len(ckpt['classes'])} classes, {dest.stat().st_size / 1e6:.1f} MB")
    return wrapped, ckpt


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    for old in ("model.onnx", "labels.json"):  # the single-model app files
        (MODELS / old).unlink(missing_ok=True)
    shutil.rmtree(REFS, ignore_errors=True)
    REFS.mkdir(parents=True)

    _, router = export("router")
    manifest = {"router": {"file": "models/router.onnx", "size": router["size"], "labels": router["classes"]},
                "specialists": {}}
    ds = datasets.ImageFolder(DATA)
    refs = []
    for crop in CROPS:
        wrapped, ckpt = export(crop)
        classes, size = ckpt["classes"], ckpt["size"]
        manifest["specialists"][crop] = {"file": f"models/{crop}.onnx", "size": ckpt["size"], "labels": classes}

        # Reference set: the most confidently-correct validation photos of each condition.
        scored = defaultdict(list)
        idx = ckpt["val_idx"]
        with torch.no_grad():
            for start in range(0, len(idx), 64):
                chunk = idx[start:start + 64]
                x = to_input(np.stack([centre(ds.loader(ds.samples[i][0]), size) for i in chunk]), "cpu")
                logits, emb = wrapped(x)
                prob = logits.softmax(1)
                for j, i in enumerate(chunk):
                    y = classes.index(ds.classes[ds.samples[i][1]])
                    # images from datasets with no stated licence are used to train but never shipped as thumbnails
                    if prob[j].argmax().item() == y and "nolic" not in Path(ds.samples[i][0]).name:
                        scored[y].append((prob[j, y].item(), i, emb[j].numpy()))
        for y, items in sorted(scored.items()):
            items.sort(key=lambda t: -t[0])
            # spread picks across the confident half rather than the very top few near-duplicates
            pool = items[: max(REFS_PER_CLASS, len(items) // 2)]
            picks = [pool[k * len(pool) // REFS_PER_CLASS] for k in range(min(REFS_PER_CLASS, len(pool)))]
            for k, (_, i, e) in enumerate(picks):
                name = f"{classes[y]}_{k}.jpg"
                thumb(ds.samples[i][0], REFS / name)
                q = np.clip(np.round(e * 127 / max(1e-6, np.abs(e).max())), -127, 127).astype(np.int8)
                refs.append({"id": classes[y], "crop": crop, "img": f"refs/{name}",
                             "emb": base64.b64encode(q.tobytes()).decode()})

    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (MODELS / "refs.json").write_text(json.dumps(refs))
    metrics = {"models": {t: json.loads((RUNS / t / "metrics.json").read_text()) for t in ["router", *CROPS]}}
    if (RUNS / "pipeline_metrics.json").exists():
        metrics["pipeline"] = json.loads((RUNS / "pipeline_metrics.json").read_text())
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote manifest, {len(refs)} reference images, metrics")


if __name__ == "__main__":
    main()

"""Export the single 28-class model (runs/all/best.pt) in the layout the app loads when it has no manifest.json:

    python export_all.py            # writes runs/all/app_files/{models,refs}; copy them into app/public to ship
    python export_all.py --install  # also replaces app/public/models/{model.onnx,labels.json,refs.json,metrics.json}
                                    # and app/public/refs/ (the old files are kept in runs/all/previous_app_files/)
"""

import argparse
import base64
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
from torchvision import datasets

from export import APP, REFS_PER_CLASS, Exported, half_weights, thumb
from train import DATA, RUNS, centre, crop_of, to_input

APP_FILES = ["models/model.onnx", "models/labels.json", "models/refs.json", "models/metrics.json", "refs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()

    run = RUNS / "all"
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    classes, size = ckpt["classes"], ckpt["size"]
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    wrapped = Exported(model).eval()

    dest = run / "app_files"
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "models").mkdir(parents=True)
    (dest / "refs").mkdir()
    onnx_path = dest / "models" / "model.onnx"
    torch.onnx.export(wrapped, torch.randn(1, 3, size, size), onnx_path, input_names=["input"],
                      output_names=["logits", "embedding"], opset_version=17, dynamo=False,
                      dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}, "embedding": {0: "n"}})
    half_weights(onnx_path)
    x = torch.randn(4, 3, size, size)
    got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(["logits"], {"input": x.numpy()})[0]
    want = wrapped(x)[0].detach().numpy()
    assert (got.argmax(1) == want.argmax(1)).all() and np.abs(got - want).max() < 0.05, "ONNX output differs"
    print(f"exported model.onnx: {size}px, {len(classes)} classes, {onnx_path.stat().st_size / 1e6:.1f} MB")
    (dest / "models" / "labels.json").write_text(json.dumps(classes))

    # Reference photos: confidently-correct validation photos of each condition, as in export.py.
    ds = datasets.ImageFolder(DATA)
    scored = defaultdict(list)
    idx = ckpt["val_idx"]
    with torch.no_grad():
        for start in range(0, len(idx), 64):
            chunk = idx[start:start + 64]
            logits, emb = wrapped(to_input(np.stack([centre(ds.loader(ds.samples[i][0]), size) for i in chunk]), "cpu"))
            prob = logits.softmax(1)
            for j, i in enumerate(chunk):
                y = classes.index(ds.classes[ds.samples[i][1]])
                if prob[j].argmax().item() == y and "nolic" not in Path(ds.samples[i][0]).name:
                    scored[y].append((prob[j, y].item(), i, emb[j].numpy()))
    refs = []
    for y, items in sorted(scored.items()):
        items.sort(key=lambda t: -t[0])
        pool = items[: max(REFS_PER_CLASS, len(items) // 2)]
        for k, (_, i, e) in enumerate(pool[n * len(pool) // REFS_PER_CLASS] for n in range(min(REFS_PER_CLASS, len(pool)))):
            name = f"{classes[y]}_{k}.jpg"
            thumb(ds.samples[i][0], dest / "refs" / name)
            q = np.clip(np.round(e * 127 / max(1e-6, np.abs(e).max())), -127, 127).astype(np.int8)
            refs.append({"id": classes[y], "crop": crop_of(classes[y]), "img": f"refs/{name}",
                         "emb": base64.b64encode(q.tobytes()).decode()})
    (dest / "models" / "refs.json").write_text(json.dumps(refs))
    m = json.loads((run / "metrics.json").read_text())
    if (run / "web_test.json").exists():
        m["web_test"] = json.loads((run / "web_test.json").read_text())
    (dest / "models" / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"wrote {len(refs)} reference photos -> {dest}")

    if args.install:
        backup = run / "previous_app_files"
        shutil.rmtree(backup, ignore_errors=True)
        for rel in APP_FILES:
            src, live = dest / rel, APP / rel
            if live.exists():
                (backup / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(live, backup / rel)
            shutil.copytree(src, live) if src.is_dir() else shutil.copy(src, live)
        print(f"installed into {APP} (previous files in {backup})")


if __name__ == "__main__":
    main()

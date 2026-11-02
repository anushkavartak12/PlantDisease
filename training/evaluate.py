"""Score the two-stage pipeline (router -> crop specialist) on the shared validation split, the same way the app
runs it, and compare with the single 28-class model (runs/all) when it exists.

    python evaluate.py            # writes runs/pipeline_metrics.json
"""

import json
from collections import defaultdict

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import datasets

from train import CROPS, DATA, RUNS, centre, crop_of, is_field, to_input

OTHER_ID = "other___unsupported"
UNSUPPORTED_AT = 0.5  # router says "other" with at least this probability: not a crop we know
SECOND_CROP_AT = 0.2  # also ask the runner-up crop's specialist when the router gives it at least this much


def load(task: str, device: str):
    ckpt = torch.load(RUNS / task / "best.pt", map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device), ckpt


def views(im: Image.Image, size: int, multi: bool) -> np.ndarray:
    """The app's views of a photo: centre square, its mirror image, and the whole photo squeezed to a square."""
    c = centre(im, size)
    vs = [c]
    if multi:
        vs += [c[:, ::-1], np.asarray(im.resize((size, size), Image.BILINEAR))]
    return np.ascontiguousarray(np.stack(vs))


@torch.no_grad()
def probs(model, im, size, multi, device) -> torch.Tensor:
    return model(to_input(views(im, size, multi), device)).float().softmax(1).mean(0).cpu()


def pipeline(im, router, specialists, multi, device) -> tuple[str, str]:
    """(predicted crop, predicted condition id), as the app decides them."""
    model, ck = router
    r = probs(model, im, ck["size"], multi, device)
    names = ck["classes"]
    if names[r.argmax()] == "other" and r.max() >= UNSUPPORTED_AT:
        return "other", OTHER_ID
    ranked = [names[k] for k in r.argsort(descending=True).tolist() if names[k] != "other"]
    best, best_p = None, -1.0
    for rank, crop in enumerate(ranked[:2]):
        p_crop = r[names.index(crop)].item()
        if rank == 1 and p_crop < SECOND_CROP_AT:
            break
        sm, sck = specialists[crop]
        s = probs(sm, im, sck["size"], multi, device)
        k = int(s.argmax())
        if p_crop * s[k].item() > best_p:
            best, best_p = sck["classes"][k], p_crop * s[k].item()
    return crop_of(best), best


def flat(im, model_ck, multi, device) -> tuple[str, str]:
    model, ck = model_ck
    p = probs(model, im, ck["size"], multi, device)
    per_crop = defaultdict(float)
    for c, v in zip(ck["classes"], p.tolist()):
        per_crop[crop_of(c)] += v
    return max(per_crop, key=per_crop.get), ck["classes"][int(p.argmax())]


def score(name, predict, samples, classes):
    hit, crop_hit, field_hit, field_crop_hit, n_field = 0, 0, 0, 0, 0
    per = defaultdict(lambda: [0, 0])
    for path, y in samples:
        truth = classes[y]
        crop, cond = predict(Image.open(path).convert("RGB"))
        ok, crop_ok = cond == truth, crop == crop_of(truth)
        hit += ok
        crop_hit += crop_ok
        per[truth][0] += ok
        per[truth][1] += 1
        if is_field(path):
            n_field += 1
            field_hit += ok
            field_crop_hit += crop_ok
    n = len(samples)
    m = {"disease_acc": hit / n, "field_disease_acc": field_hit / max(1, n_field), "crop_acc": crop_hit / n,
         "field_crop_acc": field_crop_hit / max(1, n_field), "images": n, "field_images": n_field,
         "per_class": {c: a / t for c, (a, t) in sorted(per.items())}}
    print(f"{name:34s} disease {m['disease_acc']:.4f}  field disease {m['field_disease_acc']:.4f}  "
          f"leaf type {m['crop_acc']:.4f}  field leaf type {m['field_crop_acc']:.4f}", flush=True)
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = datasets.ImageFolder(DATA)
    router = load("router", device)
    specialists = {c: load(c, device) for c in CROPS}
    samples = [base.samples[i] for i in router[1]["val_idx"]]
    for c, (_, ck) in specialists.items():  # every model must have been trained on the same split
        assert set(ck["val_idx"]) <= set(router[1]["val_idx"]), f"{c} was trained on a different split"
    print(f"{len(samples)} validation photos ({sum(is_field(p) for p, _ in samples)} field)\n")

    results = {}
    for multi in (False, True):
        tag = "multi-view" if multi else "centre crop"
        results[f"pipeline, {tag}"] = score(f"router + 5 specialists, {tag}",
                                            lambda im: pipeline(im, router, specialists, multi, device),
                                            samples, base.classes)
    if (RUNS / "all" / "best.pt").exists():
        single = load("all", device)
        if set(single[1]["val_idx"]) == set(router[1]["val_idx"]):
            for multi in (False, True):
                tag = "multi-view" if multi else "centre crop"
                results[f"single model, {tag}"] = score(f"single 28-class model, {tag}",
                                                        lambda im: flat(im, single, multi, device),
                                                        samples, base.classes)
    (RUNS / "pipeline_metrics.json").write_text(json.dumps(results, indent=2))

    best = results["pipeline, multi-view"]["per_class"]
    print("\nper class (pipeline, multi-view):")
    for c, a in best.items():
        print(f"  {c:40s} {a:.3f}")


if __name__ == "__main__":
    main()

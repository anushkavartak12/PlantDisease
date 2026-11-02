"""Score the router + crop specialists on web_test_photos/<crop>/<condition>/*.jpg, photos from the web that are kept
out of training (prepare.py drops near-copies of them). This is a harder and more honest test than the validation
split, which comes from the same datasets as the training photos.

    python webtest.py                  # prints a table, writes runs/web_test.json
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from evaluate import SECOND_CROP_AT, UNSUPPORTED_AT, load, probs
from train import CROPS, RUNS

ROOT = Path(__file__).parent.parent / "web_test_photos"


@torch.no_grad()
def predict(im, router, specialists, device):
    """(router's top crop, best condition, top-3 conditions), as the app decides them."""
    model, ck = router
    r = probs(model, im, ck["size"], True, device)
    names = ck["classes"]
    if names[r.argmax()] == "other" and r.max() >= UNSUPPORTED_AT:
        return "other", "other___unsupported", ["other___unsupported"]
    ranked = [names[k] for k in r.argsort(descending=True).tolist() if names[k] != "other"]
    cands = []
    for rank, crop in enumerate(ranked[:2]):
        p_crop = r[names.index(crop)].item()
        if rank == 1 and p_crop < SECOND_CROP_AT:
            break
        sm, sck = specialists[crop]
        s = probs(sm, im, sck["size"], True, device)
        cands += [(c, p_crop * v) for c, v in zip(sck["classes"], s.tolist())]
    cands.sort(key=lambda t: -t[1])
    return names[int(r.argmax())], cands[0][0], [c for c, _ in cands[:3]]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = load("router", device)
    specialists = {c: load(c, device) for c in CROPS}
    known = {c for _, ck in specialists.values() for c in ck["classes"]}
    rows = []
    for f in sorted(ROOT.rglob("*.jpg")):
        truth = f"{f.parent.parent.name}___{f.parent.name}"
        if truth not in known:  # e.g. a crop or condition the app no longer covers
            continue
        crop, pred, top3 = predict(Image.open(f).convert("RGB"), router, specialists, device)
        rows.append({"file": str(f.relative_to(ROOT)), "truth": truth, "pred": pred, "crop": crop,
                     "top3": truth in top3, "source": "bugwood" if f.name.startswith("bugwood_") else "other"})

    def line(name, rs):
        n = len(rs)
        if n:
            ok = sum(r["truth"] == r["pred"] for r in rs)
            crop_ok = sum(r["truth"].split("___")[0] == r["crop"] for r in rs)
            print(f"{name:34s} {ok:3d}/{n:<3d} {100 * ok / n:4.0f}%   leaf type {100 * crop_ok / n:4.0f}%   "
                  f"top-3 {100 * sum(r['top3'] for r in rs) / n:4.0f}%")

    line("all photos", rows)
    line("  Bugwood photos", [r for r in rows if r["source"] == "bugwood"])
    line("  other photos", [r for r in rows if r["source"] != "bugwood"])
    by = defaultdict(list)
    for r in rows:
        by[r["truth"].split("___")[0]].append(r)
    print()
    for crop, rs in sorted(by.items()):
        line(crop, rs)
    print()
    for r in rows:
        if r["truth"] != r["pred"]:
            print(f"  {r['truth']:32s} -> {r['pred']:32s} {r['file']}")
    (RUNS / "web_test.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

"""Cut the leaf out of every plain-background (lab) photo, so train.py can paste it onto real field backgrounds.

Writes data/masks/<class>/<image stem>.png (255 = leaf). Photos whose backdrop is not plain (grass, soil,
other leaves) and photos where the cut-out looks wrong get no mask, and train.py leaves them as they are.

    python masks.py            # after prepare.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "images"
MASKS = ROOT / "data" / "masks"
WORK = 200  # px, longest side GrabCut runs at
# Classes whose lab photos are leaves on a plain sheet or table (cotton, PlantVillage tomato and "other", white-sheet
# rice and PlantVillage soybean healthy). The other lab sets are already shot in the field, where a cut-out would
# only lose part of the leaf.
PLAIN = ("cotton___", "tomato___", "other___", "rice___healthy", "soybean___healthy")


def plain_backdrop(lab: np.ndarray) -> np.ndarray | None:
    """Median border colour (Lab) if most of the border is one plain colour, else None."""
    h, w, _ = lab.shape
    b = max(3, min(h, w) // 40)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]).astype(np.float32)
    median = np.median(border, 0)
    # A leaf can touch the edge, and backdrops have shadows and gradients that change brightness but hardly the
    # colour, so compare colour (a, b) only and only require most of the border to match. Grass and soil do not.
    near = (np.hypot(border[:, 1] - median[1], border[:, 2] - median[2]) < 9).mean()
    return median if near > 0.6 else None


def leaf_mask(bgr: np.ndarray) -> np.ndarray | None:
    full = bgr.shape[:2]
    s = WORK / max(full)
    if s < 1:  # GrabCut is slow; cut at low resolution and scale the mask back up
        bgr = cv2.resize(bgr, (round(full[1] * s), round(full[0] * s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    backdrop = plain_backdrop(lab)
    if backdrop is None:
        return None
    h, w, _ = bgr.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    # Shadows are darker versions of the backdrop: compare chroma (a, b) more strictly than lightness.
    d = lab.astype(np.float32) - backdrop
    chroma = np.hypot(d[..., 1], d[..., 2])
    bd_sat = np.median(sat[np.linalg.norm(d, axis=2) < 20]) if (np.linalg.norm(d, axis=2) < 20).any() else 0
    fg = (chroma > 14) | (sat > bd_sat + 45)
    # cast shadows: dark and grey (dark leaf tissue is still saturated)
    shadow = (hsv[..., 2] < 75) & (sat < 80)
    fg &= ~shadow

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[fg] = cv2.GC_PR_FGD
    core = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
    gc[core & (chroma > 25)] = cv2.GC_FGD
    gc[shadow & (cv2.erode(shadow.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)] = cv2.GC_BGD
    b = 3
    gc[:b], gc[-b:], gc[:, :b], gc[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    # a leaf touching the edge: let those border pixels be decided by the model rather than forced
    edge_fg = fg & (chroma > 30)
    for sl in (np.s_[:b], np.s_[-b:], np.s_[:, :b], np.s_[:, -b:]):
        gc[sl][edge_fg[sl]] = cv2.GC_PR_FGD
    if (gc == cv2.GC_PR_FGD).sum() + (gc == cv2.GC_FGD).sum() < 0.03 * h * w:
        return None
    try:
        cv2.grabCut(bgr, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

    # keep the leaf and fill its holes (lesions and insect holes stay part of the leaf)
    n, lab_ids, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    # A leaf is one solid piece: a cut split into several sizeable pieces (a striped leaf filling the frame, lesions
    # mistaken for the leaf) is wrong, so no mask.
    if areas.sum() - areas.max() > 0.12 * areas.sum():
        return None
    keep = [1 + int(areas.argmax())]
    m = np.isin(lab_ids, keep).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.drawContours(np.zeros_like(m), cnts, -1, 1, cv2.FILLED)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    frac = m.mean()
    if not 0.06 < frac < 0.9:
        return None
    # The cut-out must look like the backdrop outside and not like it inside, or it is a bad cut.
    out_far = (chroma[m == 0] > 30).mean()
    if out_far > 0.15:
        return None
    m = m * 255
    if s < 1:
        m = cv2.resize(m, (full[1], full[0]), interpolation=cv2.INTER_LINEAR)
        m = np.where(m > 127, 255, 0).astype(np.uint8)
    return m


def work(path: Path) -> bool:
    dest = MASKS / path.parent.name / (path.stem + ".png")
    if dest.exists():
        return True
    bgr = cv2.imread(str(path))
    if bgr is None:
        return False
    m = leaf_mask(bgr)
    if m is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), m)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    paths = sorted(p for p in DATA.rglob("*.jpg")
                   if not p.name.startswith("field_") and p.parent.name.startswith(PLAIN))
    print(f"{len(paths)} lab photos")
    with ProcessPoolExecutor(args.workers) as ex:
        done = list(ex.map(work, paths, chunksize=32))
    per_class: dict[str, list[int]] = {}
    for p, ok in zip(paths, done):
        per_class.setdefault(p.parent.name, [0, 0])
        per_class[p.parent.name][0] += ok
        per_class[p.parent.name][1] += 1
    for c, (ok, n) in sorted(per_class.items()):
        print(f"  {c:40s} {ok:4d} / {n:4d} masked")
    print(f"total {sum(done)} / {len(paths)} masked")


if __name__ == "__main__":
    main()

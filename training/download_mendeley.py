"""Download extra field-photo datasets (CC BY 4.0, Mendeley Data) into data/raw/<name>/, keeping their folder layout.

    python download_mendeley.py --list      # show folders and sizes only
    python download_mendeley.py             # download (resumes: existing files of the right size are skipped)
"""

import argparse
import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
API = "https://data.mendeley.com/public-api/datasets"
# name -> (dataset id, version, file names to take; None = everything)
DATASETS = {
    # MH-SoyaHealthVision, Maharashtra field photos: rust and frogeye only (healthy soybean is already plentiful)
    "soya_mh": ("hkbgh5s3b7", 1, ["Soyabean_Frog_Leaf_Eye.zip", "Soyabean_Rust.zip"]),
    "tomato_pk": ("3mbnb82mxd", 2, None),  # Tomato Leaf Disease Classification Dataset in Pakistan, field photos
    # Optional, 2 GB for ~100 photos per class: Multi-Class Soybean Leaf Disease Dataset, field photos
    # "soya_multi": ("6fhphxg297", 2, None),
}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        return json.load(r)


def walk(ds, ver, folder="root", path=()):
    """(relative path, size, download url) of every file under `folder`."""
    for f in get(f"{API}/{ds}/files?folder_id={folder}&version={ver}"):
        cd = f["content_details"]
        yield Path(*path, f["filename"]), cd["size"], cd["download_url"]
    if folder == "root":
        subs = [s for s in get(f"{API}/{ds}/folders/{ver}") if not s.get("parent_id")]
    else:
        subs = [s for s in get(f"{API}/{ds}/folders/{ver}") if s.get("parent_id") == folder]
    for s in subs:
        yield from walk(ds, ver, s["id"], (*path, s["name"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    for name, (ds, ver, keep) in DATASETS.items():
        files = [f for f in walk(ds, ver) if keep is None or f[0].name in keep]
        print(f"{name}: {len(files)} files, {sum(s for _, s, _ in files) / 1e6:.0f} MB")
        if args.list:
            for top in sorted({str(p.parent) for p, _, _ in files}):
                print("  ", top, sum(1 for p, _, _ in files if str(p.parent) == top))
            continue
        for rel, size, url in files:
            dest = RAW / name / rel
            if dest.exists() and dest.stat().st_size == size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            have = part.stat().st_size if part.exists() else 0  # resume a partial download
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                done = have if r.status == 206 else 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if done % (50 << 20) < (1 << 20):
                        print(f"  {rel}: {done / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
            part.replace(dest)
        print(f"  done -> {RAW / name}")


if __name__ == "__main__":
    main()

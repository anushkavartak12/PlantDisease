"""Round 4 field photos (fetch after download.sh and download_mendeley.py):

- potato: PLDD-UP (Uttar Pradesh, India; Mendeley 3j4nfkvp2n) and the Indonesian uncontrolled-environment set
  (Mendeley ptz377bwb8), both CC BY 4.0
- cotton: Cotton Leaf Disease Dataset with Severity Levels (Maharashtra, India; Mendeley 74jsdxtmx2), CC BY 4.0
- soybean: Multi-Class Soybean Leaf Disease Dataset (Mendeley 6fhphxg297), CC BY 4.0
- Bugwood / IPM Images photos of soybean and cotton diseases (CC BY 3.0 US, credit kept in bugwood/_credits.json)

Large zips are read with HTTP range requests, so only the sampled photos are downloaded.

    python download_extra.py
"""

import json
import random
import re
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_mendeley import RAW, walk
from remote_zip import open_zip

UA = {"User-Agent": "Mozilla/5.0 (PlantDoctor)"}
IMAGE_EXT = (".jpg", ".jpeg", ".png")
# Mendeley dataset -> (version, {zip name or folder name: (raw subfolder, max photos)})
SETS = {
    "3j4nfkvp2n": (1, {"EB.zip": ("potato_up/EB", 900), "LB.zip": ("potato_up/LB", 900),
                       "Healthy.zip": ("potato_up/Healthy", 900)}),
    "74jsdxtmx2": (3, {"Cotton-Original.zip": ("cotton_in", 0)}),  # 0: everything
    "ptz377bwb8": (1, {"Phytopthora": ("potato_id/Phytopthora", 0), "Healthy": ("potato_id/Healthy", 0)}),
    "6fhphxg297": (2, {"Bacterial Blight": ("soya_multi/Bacterial Blight", 0), "Rust": ("soya_multi/Rust", 0)}),
}
# Bugwood subject ids (host-specific pathogens) -> class folder name
BUGWOOD = {9439: "soybean_frogeye", 10499: "soybean_bacterial_blight", 9298: "soybean_rust",
           9425: "cotton_bacterial_blight"}
BUGWOOD_MAX = 80
THREADS = 6


def fetch(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def mendeley():
    for ds, (ver, wanted) in SETS.items():
        files = list(walk(ds, ver))
        for key, (sub, cap) in wanted.items():
            dest = RAW / sub
            if key.endswith(".zip"):
                url = next(u for p, _, u in files if p.name == key)
                members = [m for m in open_zip(url).infolist() if m.filename.lower().endswith(IMAGE_EXT)]
                random.Random(0).shuffle(members)
                local = threading.local()  # one open zip per thread: each has its own read position

                def get(m, url=url, dest=dest, local=local):
                    out = dest / m.filename
                    if not out.exists():
                        if not hasattr(local, "z"):
                            local.z = open_zip(url)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(local.z.read(m.filename))
                todo = members[: cap or None]
            else:
                def get(item, dest=dest):
                    p, u = item
                    out = dest / p.name
                    if not out.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(fetch(u))
                todo = [(p, u) for p, _, u in files if p.parent.name == key and p.suffix.lower() in IMAGE_EXT][: cap or None]
            with ThreadPoolExecutor(THREADS) as ex:  # the link is latency-bound: parallel requests are ~3x faster
                list(ex.map(get, todo))
            print(f"{ds}/{key}: {len(todo)} photos -> {dest}", flush=True)


def bugwood():
    credits = []
    for sub, name in BUGWOOD.items():
        rows = json.loads(fetch(f"https://api.bugwood.org/rest/api/image/.json?sub={sub}&rows=200"))["rows"]
        kept = 0
        for r in rows:
            if kept >= BUGWOOD_MAX:
                break
            i = r["imgnum"]
            # skip microscope slides, spores and lab cultures: the app only sees leaves
            page = fetch(f"https://www.ipmimages.org/browse/image/{i}").decode("utf-8", "replace")
            m = re.search(r'\\"imagetype\\":\\"(.*?)\\"', page)
            if m and re.search(r"micro|spore|culture|sign", m.group(1), re.I):
                continue
            out = RAW / "bugwood" / name / f"bugwood_{i}.jpg"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(fetch(f"https://bugwoodcloud.org/images/768x512/{i}.jpg"))
            credits.append({"file": f"{name}/{out.name}", "source": f"https://www.ipmimages.org/browse/image/{i}",
                            "credit": f'{r.get("photographer", "").strip()}, {r.get("organization", "")}, Bugwood.org'})
            kept += 1
        print(f"bugwood {name}: {kept} photos", flush=True)
    (RAW / "bugwood" / "_credits.json").write_text(json.dumps(credits, indent=1))


if __name__ == "__main__":
    mendeley()
    bugwood()

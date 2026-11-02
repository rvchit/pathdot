"""Collect openly licensed, captioned histopathology images from Wikimedia Commons.

Only keeps CC0 / public domain / CC BY / CC BY-SA files, and records license, author
and source URL for attribution.

    python scripts/source_images.py --out data --max 1500
Output: data/images/commons/<pageid>.jpg and data/commons_images.jsonl
"""
import argparse, html, io, json, re, time
from pathlib import Path
import requests
from PIL import Image
from common import save_jpeg

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "pathvqa-hackathon/0.1 (research; contact: your-email@example.com)"}
OK_LICENSES = re.compile(r"^(cc0|public domain|pd|cc[- ]by(-sa)?([- ]\d\.\d)?)", re.I)

CATEGORIES = [
    "Histopathology", "Micrographs of tumors", "Histology of neoplasms",
    "Histopathology of the liver", "Histopathology of the lung",
    "Histopathology of the kidney", "Histopathology of the breast",
    "Histopathology of the colon", "Histopathology of the skin",
    "Histopathology of the thyroid", "H&E stained micrographs",
]
SEARCHES = [
    "histopathology micrograph H&E", "H&E stain carcinoma micrograph",
    "histology lymph node micrograph", "adenocarcinoma histopathology",
    "granuloma histopathology", "necrosis histopathology micrograph",
]


def get(params):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(5):
        r = requests.get(API, params=params, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    r.raise_for_status()


def files_in_category(cat, depth=1, seen_cats=None):
    seen_cats = seen_cats if seen_cats is not None else set()
    if cat in seen_cats:
        return
    seen_cats.add(cat)
    cont = {}
    while True:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                 "cmtype": "file|subcat", "cmlimit": 500, **cont})
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 6:
                yield m["title"]
            elif m["ns"] == 14 and depth > 0:
                yield from files_in_category(m["title"].split(":", 1)[1], depth - 1, seen_cats)
        if "continue" not in d:
            break
        cont = d["continue"]


def files_from_search(q, limit=500):
    d = get({"action": "query", "list": "search", "srsearch": f"{q} filetype:bitmap",
             "srnamespace": 6, "srlimit": limit})
    for s in d["query"]["search"]:
        yield s["title"]


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).strip()


def info(titles):
    d = get({"action": "query", "prop": "imageinfo", "titles": "|".join(titles),
             "iiprop": "url|extmetadata|mime", "iiurlwidth": 768})
    for p in d["query"]["pages"]:
        if "imageinfo" not in p:
            continue
        ii = p["imageinfo"][0]
        md = ii.get("extmetadata", {})
        yield {
            "pageid": p["pageid"], "title": p["title"], "mime": ii.get("mime"),
            "thumb": ii.get("thumburl") or ii.get("url"), "page_url": ii.get("descriptionurl"),
            "license": md.get("LicenseShortName", {}).get("value", ""),
            "artist": strip_html(md.get("Artist", {}).get("value", "")),
            "caption": strip_html(md.get("ImageDescription", {}).get("value", "")),
            "categories": md.get("Categories", {}).get("value", ""),
        }


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--max", type=int, default=1500)
ap.add_argument("--min-caption-words", type=int, default=6)
args = ap.parse_args()
out = Path(args.out)
img_dir = out / "images" / "commons"

titles = []
for c in CATEGORIES:
    try:
        titles += list(files_in_category(c))
    except Exception as e:
        print("category failed", c, e)
for q in SEARCHES:
    titles += list(files_from_search(q))
titles = list(dict.fromkeys(titles))
print(f"{len(titles)} candidate files")

kept, out_f = 0, open(out / "commons_images.jsonl", "w")
for i in range(0, len(titles), 50):
    if kept >= args.max:
        break
    for m in info(titles[i:i + 50]):
        if m["mime"] not in ("image/jpeg", "image/png", "image/tiff"):
            continue
        if not OK_LICENSES.match(m["license"].strip()):
            continue
        if len(m["caption"].split()) < args.min_caption_words:
            continue
        try:
            r = requests.get(m["thumb"], headers=UA, timeout=60)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
        except Exception as e:
            print("download failed", m["title"], e)
            continue
        p = img_dir / f"{m['pageid']}.jpg"
        save_jpeg(img, p)
        m.update(image_path=str(p), source="wikimedia_commons")
        out_f.write(json.dumps(m) + "\n")
        kept += 1
        if kept >= args.max:
            break
    print(f"kept {kept}")
    time.sleep(0.5)
out_f.close()
print(f"done: {kept} images -> {out/'commons_images.jsonl'}")

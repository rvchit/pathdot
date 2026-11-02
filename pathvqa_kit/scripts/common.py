"""Shared helpers: image normalization, hashing, prompt template, answer normalization."""
import base64, io, json, re
from pathlib import Path
from PIL import Image

MAX_SIDE = 512          # same size for training, eval and every model
JPEG_QUALITY = 90

SYSTEM_PROMPT = (
    "You are a pathology assistant. Look at the image and answer the question. "
    "If it is a yes/no question, answer exactly 'yes' or 'no'. "
    "Otherwise answer with as few words as possible (usually 1-3 words). "
    "Output only the answer."
)


def normalize_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return img


def save_jpeg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_image(img).save(path, "JPEG", quality=JPEG_QUALITY)


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def user_content(image_path: Path, question: str) -> list:
    return [
        {"type": "image_url", "image_url": {"url": data_uri(image_path)}},
        {"type": "text", "text": question},
    ]


def norm_answer(a: str) -> str:
    a = str(a).lower().strip()
    a = re.sub(r"[^\w\s]", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def is_closed(answer: str) -> bool:
    return norm_answer(answer) in {"yes", "no"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

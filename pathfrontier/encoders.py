"""Tile encoders (CONTRACTS.md section 2).

A registry of pathology foundation models plus two non-pathology controls, each wrapped in an
`Encoder` with a uniform `embed(list[PIL.Image]) -> (N, dim) float32` interface.

Loader facts below were checked against each Hugging Face model card (README + config.json)
on 2026-10-29; the URL is cited next to every spec.  Gated repos need `HF_TOKEN` (see
`pathfrontier.config`) and prior access approval on the model page.
"""
from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image

from pathfrontier import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LoaderKind = Literal["hf_automodel", "timm_hub", "timm_bin", "timm_pretrained", "conch", "fake"]


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hub_id: str                      # Hugging Face repo id ("" for synthetic encoders)
    loader: LoaderKind
    input_size: int                  # square side the tile is resized to before the forward pass
    dim: int                         # embedding dimension returned by `Encoder.embed`
    gated: bool
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    timm_arch: str = ""              # architecture name for timm_bin / timm_pretrained loaders
    timm_kwargs: dict = field(default_factory=dict)
    pooling: Literal["cls", "cls_mean_patch", "pooled", "conch"] = "cls"
    patch_start: int = 1             # first patch-token index (after CLS + register tokens)
    probe_dim: bool = False          # dim not verifiable offline: measure it with a dummy forward at load
    note: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.hub_id}" if self.hub_id else ""


ENCODERS: dict[str, EncoderSpec] = {
    # https://huggingface.co/owkin/phikon-v2 : `AutoModel.from_pretrained("owkin/phikon-v2")`, feature =
    # `outputs.last_hidden_state[:, 0, :]` (CLS) of shape (1, 1024). preprocessor_config.json: resize
    # shortest edge 224, center-crop 224, ImageNet mean/std. Dinov2Model, ViT-L/16, 224 input. Ungated.
    "phikon-v2": EncoderSpec(
        name="phikon-v2", hub_id="owkin/phikon-v2", loader="hf_automodel", input_size=224,
        dim=1024, gated=False, pooling="cls",
        note="Owkin Phikon-v2 (ViT-L/16 DINOv2, PANCAN-XL). Ungated, Owkin non-commercial licence.",
    ),
    # https://huggingface.co/MahmoodLab/UNI : option 2 of the model card downloads
    # `pytorch_model.bin` and builds `timm.create_model("vit_large_patch16_224", img_size=224,
    # patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)`; transform = resize
    # 224 + ImageNet mean/std; output shape [1, 1024]. Gated (institutional email required).
    "uni": EncoderSpec(
        name="uni", hub_id="MahmoodLab/UNI", loader="timm_bin", input_size=224, dim=1024,
        gated=True, timm_arch="vit_large_patch16_224",
        timm_kwargs={"img_size": 224, "patch_size": 16, "init_values": 1e-5, "num_classes": 0,
                         "dynamic_img_size": True},
        pooling="pooled",
        note="Mahmood Lab UNI (ViT-L/16, Mass-100K). Gated, CC-BY-NC-ND 4.0.",
    ),
    # https://huggingface.co/MahmoodLab/UNI2-h : `timm_kwargs = {img_size: 224, patch_size: 14,
    # depth: 24, num_heads: 24, init_values: 1e-5, embed_dim: 1536, mlp_ratio: 2.66667*2,
    # num_classes: 0, no_embed_class: True, mlp_layer: timm.layers.SwiGLUPacked,
    # act_layer: torch.nn.SiLU, reg_tokens: 8, dynamic_img_size: True}` on
    # `vit_giant_patch14_224`, weights `pytorch_model.bin`, ImageNet mean/std, output [1, 1536]. Gated.
    "uni2-h": EncoderSpec(
        name="uni2-h", hub_id="MahmoodLab/UNI2-h", loader="timm_bin", input_size=224, dim=1536,
        gated=True, timm_arch="vit_giant_patch14_224",
        timm_kwargs={"img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
                         "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
                         "mlp_layer": "SwiGLUPacked", "act_layer": "SiLU", "reg_tokens": 8,
                         "dynamic_img_size": True},
        pooling="pooled",
        note="Mahmood Lab UNI2-h (ViT-h/14 with 8 register tokens). Gated, CC-BY-NC-ND 4.0.",
    ),
    # https://huggingface.co/MahmoodLab/CONCH : `pip install git+https://github.com/Mahmoodlab/CONCH.git`,
    # `from conch.open_clip_custom import create_model_from_pretrained`;
    # `model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
    # hf_auth_token=...)`; image features for linear probes / MIL =
    # `model.encode_image(image, proj_contrast=False, normalize=False)` (ViT-B/16 width 768,
    # 224 input, CLIP-style mean/std from the returned `preprocess`). Pre-projection width is 768
    # (post-projection would be 512); the dim is re-measured at load time to be safe. Gated.
    "conch": EncoderSpec(
        name="conch", hub_id="MahmoodLab/CONCH", loader="conch", input_size=224, dim=768,
        gated=True, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711),
        pooling="conch", probe_dim=True,
        note="Mahmood Lab CONCH vision tower (ViT-B/16). Needs `pip install "
             "git+https://github.com/Mahmoodlab/CONCH.git`. Gated, CC-BY-NC-ND 4.0.",
    ),
    # https://huggingface.co/paige-ai/Virchow2 : `timm.create_model("hf-hub:paige-ai/Virchow2",
    # pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)` ("need to specify MLP layer
    # and activation function for proper init"); transform from `resolve_data_config(model.pretrained_cfg)`
    # (224x224); output 1 x 261 x 1280: token 0 = CLS, tokens 1-4 = register tokens (ignored),
    # tokens 5: = 256 patch tokens; embedding = concat(CLS, mean(patch tokens)) -> 2560. Gated.
    "virchow2": EncoderSpec(
        name="virchow2", hub_id="paige-ai/Virchow2", loader="timm_hub", input_size=224, dim=2560,
        gated=True, timm_kwargs={"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"},
        pooling="cls_mean_patch", patch_start=5,
        note="Paige Virchow2 (ViT-H/14, 4 register tokens). Gated, CC-BY-NC-ND 4.0.",
    ),
    # https://huggingface.co/bioptimus/H-optimus-0 : `timm.create_model("hf-hub:bioptimus/H-optimus-0",
    # pretrained=True, init_values=1e-5, dynamic_img_size=False)`; transform = ToTensor + Normalize(
    # mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)); expects 224x224 at
    # 0.5 mpp; fp16 autocast on cuda; `features.shape == (1, 1536)`. Gated.
    "h-optimus-0": EncoderSpec(
        name="h-optimus-0", hub_id="bioptimus/H-optimus-0", loader="timm_hub", input_size=224,
        dim=1536, gated=True, mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517),
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True}, pooling="pooled",
        note="Bioptimus H-optimus-0 (ViT-g/14, 4 register tokens). Gated, Apache-2.0.",
    ),
    # https://huggingface.co/facebook/dinov2-base : `AutoModel.from_pretrained('facebook/dinov2-base')`,
    # CLS = `last_hidden_state[:, 0]`; config hidden_size 768; preprocessor: resize 256, crop 224,
    # ImageNet mean/std. Ungated, non-pathology control.
    "dinov2-base": EncoderSpec(
        name="dinov2-base", hub_id="facebook/dinov2-base", loader="hf_automodel", input_size=224,
        dim=768, gated=False, pooling="cls",
        note="Meta DINOv2 ViT-B/14, natural-image control (ungated).",
    ),
    # https://huggingface.co/timm/resnet50.a1_in1k : `timm.create_model('resnet50.a1_in1k',
    # pretrained=True, num_classes=0)` returns the globally pooled (batch, 2048) embedding;
    # config.json: input 224, ImageNet mean/std. Ungated, ImageNet control.
    "resnet50": EncoderSpec(
        name="resnet50", hub_id="timm/resnet50.a1_in1k", loader="timm_pretrained", input_size=224,
        dim=2048, gated=False, timm_arch="resnet50.a1_in1k", timm_kwargs={"num_classes": 0},
        pooling="pooled",
        note="timm ResNet-50 ImageNet-1k, CNN control (ungated).",
    ),
    # Synthetic encoder for tests/CI: random projection of 16x16 downsampled pixels. No download.
    "fake": EncoderSpec(
        name="fake", hub_id="", loader="fake", input_size=16, dim=64, gated=False, pooling="pooled",
        note="Offline test encoder: fixed random projection of 16x16x3 pixels to 64 dims.",
    ),
}

GATED_FALLBACK = "phikon-v2"


class GatedModelError(RuntimeError):
    """Raised when a gated Hugging Face repo rejects the download (401/403 or no token)."""


def _gated_message(spec: EncoderSpec, cause: BaseException | None = None) -> str:
    token_state = "HF_TOKEN is set but was rejected" if config.HF_TOKEN else "HF_TOKEN is not set"
    msg = (
        f"Encoder '{spec.name}' ({spec.hub_id}) is gated and the download was refused "
        f"({token_state}).\n"
        f"  1. Request access at {spec.url} (log in, accept the licence; approval can take a while).\n"
        f"  2. Create a token at https://huggingface.co/settings/tokens and put HF_TOKEN=... in .env\n"
        f"     (or export HF_TOKEN) and re-run.\n"
        f"  Ungated fallback: --encoder {GATED_FALLBACK}"
    )
    if cause is not None:
        msg += f"\n  Underlying error: {type(cause).__name__}: {str(cause).splitlines()[0][:200]}"
    return msg


def _looks_like_auth_error(exc: BaseException) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover
        GatedRepoError = HfHubHTTPError = RepositoryNotFoundError = ()  # type: ignore[assignment]
    chain: list[BaseException] = []
    e: BaseException | None = exc
    while e is not None and e not in chain:
        chain.append(e)
        e = e.__cause__ or e.__context__
    for e in chain:
        if isinstance(e, (GatedRepoError, RepositoryNotFoundError)):
            return True
        if isinstance(e, HfHubHTTPError):
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                return True
        text = str(e).lower()
        if any(k in text for k in ("401", "403", "gated", "restricted", "unauthorized",
                                   "authentication", "access to model")):
            return True
    return False


def _export_token() -> None:
    """huggingface_hub / timm / transformers all honour the HF_TOKEN env var."""
    if config.HF_TOKEN and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = config.HF_TOKEN


def _resolve_timm_kwargs(kwargs: dict) -> dict:
    """Map the string placeholders in the registry to real timm/torch classes (lazy import)."""
    import timm.layers
    import torch

    out = dict(kwargs)
    if out.get("mlp_layer") == "SwiGLUPacked":
        out["mlp_layer"] = timm.layers.SwiGLUPacked
    if out.get("act_layer") == "SiLU":
        out["act_layer"] = torch.nn.SiLU
    return out


# --------------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------------

class Encoder:
    """Uniform wrapper: `embed(list[PIL.Image]) -> (N, dim) float32` (not L2-normalised)."""

    def __init__(self, spec: EncoderSpec, device: str, model=None,
                 forward: Callable | None = None) -> None:
        self.spec = spec
        self.name = spec.name
        self.dim = spec.dim
        self.device = device
        self.input_size = spec.input_size
        self._model = model
        self._forward = forward
        self._mean = np.asarray(spec.mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.asarray(spec.std, dtype=np.float32).reshape(1, 3, 1, 1)

    # -- preprocessing ------------------------------------------------------------------------
    def preprocess(self, images: list[Image.Image]) -> np.ndarray:
        """Resize to input_size (bicubic), scale to [0,1], normalise. Returns (N,3,H,W) float32."""
        s = self.input_size
        arrs = []
        for im in images:
            if im.mode != "RGB":
                im = im.convert("RGB")
            if im.size != (s, s):
                im = im.resize((s, s), Image.BICUBIC)
            arrs.append(np.asarray(im, dtype=np.uint8))
        x = np.stack(arrs).astype(np.float32) / 255.0          # (N,H,W,3)
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))      # (N,3,H,W)
        return (x - self._mean) / self._std

    def _probe_dim(self) -> int:
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(self.preprocess([Image.new("RGB", (self.input_size,) * 2)]))
            return int(self._forward(x.to(self.device)).shape[-1])

    # -- inference -----------------------------------------------------------------------------
    def embed(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        import torch

        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        assert self._forward is not None, "Encoder has no forward function"
        out: list[np.ndarray] = []
        # fp16 autocast on cuda; plain fp32 on mps/cpu (MPS half precision is slower + less exact)
        amp = (torch.autocast(device_type="cuda", dtype=torch.float16) if self.device == "cuda"
               else contextlib.nullcontext())
        with torch.inference_mode(), amp:
            for i in range(0, len(images), batch_size):
                x = torch.from_numpy(self.preprocess(images[i:i + batch_size])).to(self.device)
                out.append(self._forward(x).float().cpu().numpy())
        emb = np.concatenate(out, axis=0).astype(np.float32, copy=False)
        if emb.shape[1] != self.dim:
            raise RuntimeError(f"{self.name}: expected dim {self.dim}, got {emb.shape[1]}")
        return emb


class FakeEncoder(Encoder):
    """Deterministic, download-free encoder for tests and CI (random projection of pixels)."""

    def __init__(self, spec: EncoderSpec, device: str = "cpu", seed: int = 0) -> None:
        super().__init__(spec, "cpu")  # pure numpy: the device argument is irrelevant
        rng = np.random.default_rng(seed)
        n_in = 3 * spec.input_size * spec.input_size
        self._proj = (rng.standard_normal((n_in, spec.dim)) / np.sqrt(n_in)).astype(np.float32)

    def embed(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(images), batch_size):
            x = self.preprocess(images[i:i + batch_size]).reshape(-1, self._proj.shape[0])
            out.append(x @ self._proj)
        return np.concatenate(out, axis=0).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------

def _load_hf_automodel(spec: EncoderSpec, device: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(spec.hub_id, token=config.HF_TOKEN or None)
    model.eval().to(device)

    def forward(x):
        return model(pixel_values=x).last_hidden_state[:, 0, :]   # CLS token

    return model, forward


def _timm_pool(spec: EncoderSpec, model):
    """Build the forward fn for a timm model given the spec's pooling rule."""
    if spec.pooling == "pooled":
        return lambda x: model(x)
    if spec.pooling == "cls":
        return lambda x: model.forward_features(x)[:, 0]
    if spec.pooling == "cls_mean_patch":
        p0 = spec.patch_start

        def forward(x):
            import torch

            tokens = model.forward_features(x)                     # (N, 1+reg+patches, C)
            return torch.cat([tokens[:, 0], tokens[:, p0:].mean(dim=1)], dim=-1)

        return forward
    raise ValueError(f"unknown pooling {spec.pooling}")


def _load_timm_hub(spec: EncoderSpec, device: str):
    import timm

    model = timm.create_model(f"hf-hub:{spec.hub_id}", pretrained=True,
                              **_resolve_timm_kwargs(spec.timm_kwargs))
    model.eval().to(device)
    return model, _timm_pool(spec, model)


def _load_timm_bin(spec: EncoderSpec, device: str):
    """Model-card 'option 2': download pytorch_model.bin and load into a timm architecture."""
    import timm
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(spec.hub_id, filename="pytorch_model.bin",
                           token=config.HF_TOKEN or None)
    model = timm.create_model(spec.timm_arch, pretrained=False,
                              **_resolve_timm_kwargs(spec.timm_kwargs))
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, _timm_pool(spec, model)


def _load_timm_pretrained(spec: EncoderSpec, device: str):
    import timm

    model = timm.create_model(spec.timm_arch, pretrained=True,
                              **_resolve_timm_kwargs(spec.timm_kwargs))
    model.eval().to(device)
    return model, _timm_pool(spec, model)


def _load_conch(spec: EncoderSpec, device: str):
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as e:
        raise ImportError(
            "CONCH needs the `conch` package: run\n"
            '  .venv/bin/pip install "git+https://github.com/Mahmoodlab/CONCH.git"\n'
            f"then request access at {spec.url} and set HF_TOKEN. "
            f"Ungated fallback: --encoder {GATED_FALLBACK}"
        ) from e

    model, _preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=config.HF_TOKEN or None)
    model.eval().to(device)

    def forward(x):
        # Pre-projection, un-normalised image features (model card: for linear probes / MIL).
        return model.encode_image(x, proj_contrast=False, normalize=False)

    return model, forward


_LOADERS: dict[str, Callable] = {
    "hf_automodel": _load_hf_automodel,
    "timm_hub": _load_timm_hub,
    "timm_bin": _load_timm_bin,
    "timm_pretrained": _load_timm_pretrained,
    "conch": _load_conch,
}


def list_encoders(include_fake: bool = False) -> list[str]:
    return [n for n in ENCODERS if include_fake or n != "fake"]


def get_encoder(name: str, device: str | None = None) -> Encoder:
    """Instantiate an encoder from the registry, downloading weights on first use."""
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder '{name}'; choose from {list(ENCODERS)}")
    spec = ENCODERS[name]
    device = device or config.device()
    if spec.loader == "fake":
        return FakeEncoder(spec, device)
    _export_token()
    try:
        model, forward = _LOADERS[spec.loader](spec, device)
    except ImportError:
        raise
    except Exception as e:
        if spec.gated and _looks_like_auth_error(e):
            raise GatedModelError(_gated_message(spec, e)) from e
        raise
    enc = Encoder(spec, device, model=model, forward=forward)
    if spec.probe_dim:
        enc.dim = int(enc._probe_dim())
    return enc


# --------------------------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------------------------

def _noise_images(n: int, size: int, seed: int = 0) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)
    return [Image.fromarray(a) for a in arr]


def benchmark(encoder: Encoder, n: int = 256, batch_size: int = 64, warmup: int = 16,
              tile_px: int = 96) -> float:
    """Warm up, then measure sustained throughput in tiles/s on `n` random `tile_px` tiles."""
    encoder.embed(_noise_images(min(warmup, n), tile_px, seed=1), batch_size=batch_size)
    _sync(encoder.device)
    images = _noise_images(n, tile_px, seed=2)
    t0 = time.perf_counter()
    encoder.embed(images, batch_size=batch_size)
    _sync(encoder.device)
    seconds = time.perf_counter() - t0
    return n / max(seconds, 1e-9)


def _sync(device: str) -> None:
    try:
        import torch

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001, S110 - sync/cache are best-effort
        pass

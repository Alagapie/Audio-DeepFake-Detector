"""
Automated download, audit, remap, and validation pipeline for AASIST pretrained weights.

Downloads the official AASIST.pth from clovaai/aasist GitHub, audits the state dict keys,
remaps them to match our AASISTModel architecture, validates shape compatibility,
and saves a clean weights file ready for inference and ONNX export.

Usage:
    python -m app.utils.download_aasist
"""

import io
import logging
from pathlib import Path

import requests
import torch

from app.config import settings

logger = logging.getLogger(__name__)

OFFICIAL_URL = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
OUTPUT_PATH = settings.weights_dir / "aasist_backbone.pt"


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------

def download_official_weights(url: str = OFFICIAL_URL, timeout: int = 120) -> bytes:
    logger.info(f"Downloading official AASIST weights from:\n  {url}")
    resp = requests.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    size_mb = len(resp.content) / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.2f} MB")
    return resp.content


# ---------------------------------------------------------------------------
# Step 2 — Audit
# ---------------------------------------------------------------------------

def audit_state_dict(state: dict) -> list[str]:
    keys = sorted(state.keys())
    print("\n" + "=" * 80)
    print("OFFICIAL STATE DICT — KEY STRUCTURE AUDIT")
    print("=" * 80)
    for i, k in enumerate(keys):
        v = state[k]
        shape_str = str(list(v.shape)) if hasattr(v, "shape") else "scalar"
        print(f"  [{i:3d}] {k:55s} {shape_str:20s} {v.dtype}")
    print(f"\n  Total keys: {len(keys)}")
    print("=" * 80 + "\n")
    return keys


# ---------------------------------------------------------------------------
# Step 3 — Remapping
# ---------------------------------------------------------------------------

def build_remap_table(official_keys: list[str]) -> dict[str, str]:
    """
    Build mapping: official_key -> our_key.

    Known structural differences:
        - Encoder: official wraps each block in nn.Sequential, ours does not.
          Pattern: encoder.X.0.YYY -> encoder.X.YYY   (remove the extra '0' index)
        - DataParallel wrapper: module.XXX.YYY -> XXX.YYY (strip prefix)
        - All other component names match one-to-one (GAT_layer_S, pool_S, etc.)
    """
    remap = {}
    for k in official_keys:
        target = k

        # Strip DataParallel wrapper if present
        if target.startswith("module."):
            target = target[7:]

        # Strip inner Sequential wrapper in encoder blocks
        # encoder.X.0.YYY -> encoder.X.YYY
        parts = target.split(".")
        if len(parts) >= 3 and parts[0] == "encoder":
            # parts = ["encoder", block_idx, "0", field, ...]
            if parts[2] == "0":
                target = ".".join(parts[:2] + parts[3:])

        remap[k] = target
    return remap


def print_remap_table(remap: dict[str, str]) -> None:
    print("\n" + "-" * 80)
    print("KEY REMAPPING TABLE")
    print("-" * 80)
    for official, ours in sorted(remap.items()):
        if official != ours:
            print(f"  {official:55s}  ->  {ours}")
    unchanged = sum(1 for o, u in remap.items() if o == u)
    print(f"\n  Unchanged: {unchanged} keys")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Step 4 — Shape Validation
# ---------------------------------------------------------------------------

def load_our_model() -> torch.nn.Module:
    from app.models.aasist_pipeline import AASISTModel
    model = AASISTModel()
    return model


def validate_shapes(
    official_state: dict,
    remap: dict[str, str],
    our_model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Validate shape compatibility for every remapped key.
    Returns the sanitized state dict with only matching keys.
    """
    our_keys = set(our_model.state_dict().keys())
    sanitized = {}
    errors = []

    print("\n" + "-" * 80)
    print("SHAPE VALIDATION")
    print("-" * 80)

    for official_key, our_key in sorted(remap.items()):
        if our_key not in our_keys:
            errors.append(f"  MISSING from our model: {our_key}  (official: {official_key})")
            continue

        official_t = official_state[official_key]
        our_t = our_model.state_dict()[our_key]

        if official_t.shape != our_t.shape:
            errors.append(
                f"  SHAPE MISMATCH: {our_key}\n"
                f"    official: {list(official_t.shape)}\n"
                f"    our:      {list(our_t.shape)}"
            )
            continue

        sanitized[our_key] = official_t
        print(f"  ✓  {our_key:55s} {list(official_t.shape)}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(e)
        raise ValueError(f"Validation failed with {len(errors)} error(s)")

    print(f"\n  All {len(sanitized)} keys validated successfully.")
    print("=" * 80 + "\n")
    return sanitized


def _check_for_lfs_pointer(data: bytes) -> bool:
    """Check if downloaded content is a Git LFS pointer file instead of actual weights."""
    try:
        text = data[:512].decode("utf-8")
        if "version https://git-lfs.github.com/spec" in text or "oid sha256:" in text:
            return True
    except UnicodeDecodeError:
        pass
    return False


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    download: bool = True,
    data: bytes | None = None,
    save: bool = True,
) -> dict[str, torch.Tensor]:
    logger.info("=" * 60)
    logger.info("AASIST WEIGHT REMAPPING PIPELINE")
    logger.info("=" * 60)

    # Step 1 — Download
    if data is None:
        if download:
            data = download_official_weights()
        else:
            raise ValueError("No data provided and download=False")

    # Check for LFS pointer
    if _check_for_lfs_pointer(data):
        logger.error(
            "Downloaded content appears to be a Git LFS pointer file.\n"
            "The raw GitHub URL returns a pointer, not actual weights.\n"
            "Alternative download options:\n"
            f"  1. Clone the repo with LFS: git lfs clone https://github.com/clovaai/aasist.git\n"
            f"     then copy models/weights/AASIST.pth to {settings.aasist_pytorch_path}\n"
            f"  2. Download from a mirror or community-hosted link\n"
        )
        raise RuntimeError("Git LFS pointer detected — cannot load weights directly from raw URL")

    # Step 2 — Load state dict
    logger.info("Loading official state dict...")
    official_state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    logger.info(f"Loaded state dict with {len(official_state)} keys")

    # Step 3 — Audit
    official_keys = audit_state_dict(official_state)

    # Step 4 — Build remap table
    remap = build_remap_table(official_keys)
    print_remap_table(remap)

    # Step 5 — Load our model architecture
    our_model = load_our_model()

    # Step 6 — Validate shapes and build sanitized state dict
    sanitized = validate_shapes(official_state, remap, our_model)

    # Step 7 — Save (optional)
    if save:
        settings.weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sanitized, OUTPUT_PATH)
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Saved remapped weights to {OUTPUT_PATH} ({size_mb:.2f} MB)")
        logger.info("Ready for inference and ONNX export.")

    return sanitized


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_pipeline(download=True, save=True)


if __name__ == "__main__":
    main()

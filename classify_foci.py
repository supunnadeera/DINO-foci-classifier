"""
DINO Foci Classifier — Standalone inference script.

Takes a cell image + segmentation mask, extracts per-cell patches,
and classifies each cell as Healthy (0) or Damaged (1) using
a DINO-pretrained ViT with gated-attention pooling head.

Usage:
    python classify_foci.py \
        --image_path  path/to/image.tif \
        --mask_path   path/to/image_mask.png \
        --model_path  models/DINO/foci_classifier.pth \
        [--backbone_name vit_tiny_patch16_384] \
        [--image_size 384] \
        [--threshold 0.5] \
        [--use_gpu]

Output (stdout): JSON with per-cell classification results.
"""

import argparse
import json
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# timm is required for the ViT backbone
try:
    import timm
except ImportError:
    print(json.dumps({"error": "timm package not installed. Run: pip install timm"}),
          file=sys.stdout)
    sys.exit(1)


# ─── Model Definition ────────────────────────────────────────────────────────

class FociViT(nn.Module):
    """DINO ViT with gated-attention MIL pooling for binary foci classification."""

    def __init__(self, backbone_name="vit_tiny_patch16_384", img_size=384):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
            num_classes=0, global_pool="", img_size=img_size
        )

        with torch.no_grad():
            dim = self.backbone.forward_features(
                torch.zeros(1, 3, img_size, img_size)
            ).shape[-1]

        hidden_dim = 128
        self.head_norm   = nn.LayerNorm(dim)
        self.attention_V = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            feats = feats["x"]
        patch_tokens = feats[:, 1:, :]  # skip CLS token

        normed = self.head_norm(patch_tokens)
        A_V    = self.attention_V(normed)
        A_U    = self.attention_U(normed)
        A_raw  = self.attention_w(A_V * A_U).squeeze(-1)
        A      = torch.softmax(A_raw, dim=1)

        pooled = torch.sum(patch_tokens * A.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        probs  = torch.sigmoid(logits)
        return probs, A


# ─── Preprocessing ────────────────────────────────────────────────────────────

NORM_MEAN = [0.2251, 0.2251, 0.2251]
NORM_STD  = [0.2375, 0.2375, 0.2375]


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


# ─── Cell Patch Extraction ───────────────────────────────────────────────────

def extract_cell_patches(image, mask_array):
    """Extract individual cell image patches from the original image using
    the segmentation mask bounding boxes.

    Returns list of (label, PIL.Image patch, bbox dict).
    """
    labels = np.unique(mask_array)
    labels = labels[labels > 0]  # skip background

    patches = []
    img_w, img_h = image.size  # PIL gives (width, height)

    for label in labels:
        ys, xs = np.where(mask_array == label)
        if len(ys) == 0:
            continue

        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        # Add padding (10% of bbox size, clamped to image bounds)
        pad_x = max(int((max_x - min_x) * 0.1), 2)
        pad_y = max(int((max_y - min_y) * 0.1), 2)
        min_x = max(0, min_x - pad_x)
        min_y = max(0, min_y - pad_y)
        max_x = min(img_w - 1, max_x + pad_x)
        max_y = min(img_h - 1, max_y + pad_y)

        patch = image.crop((min_x, min_y, max_x + 1, max_y + 1))

        bbox = {
            "min_x": min_x, "min_y": min_y,
            "max_x": max_x, "max_y": max_y
        }
        patches.append((int(label), patch, bbox))

    return patches


# ─── Channel Extraction ───────────────────────────────────────────────────────

def normalize_to_uint8(channel_data):
    """Normalize any-bit-depth channel data to uint8 [0, 255].

    Uses percentile-based contrast stretching (0.5th–99.5th percentile)
    to handle microscopy images with wide dynamic range.
    """
    arr = channel_data.astype(np.float64)

    # Percentile-based normalization for better contrast
    p_low = np.percentile(arr, 0.5)
    p_high = np.percentile(arr, 99.5)

    if p_high - p_low < 1e-6:
        # Flat image — avoid division by zero
        return np.zeros_like(channel_data, dtype=np.uint8)

    arr = (arr - p_low) / (p_high - p_low)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def extract_foci_channel(raw_image, foci_channel, image_path):
    """Extract the foci channel from a multi-channel image and return as RGB.

    For multi-channel TIFFs (e.g. 2-channel microscopy), channels are often
    stored as separate frames (pages) in the TIFF file. The channel data may
    be 16-bit or higher, so it is normalized to 8-bit [0, 255] before creating
    the RGB image.

    Args:
        raw_image: PIL Image opened from the file.
        foci_channel: 0-based channel index, or -1 for auto/default.
        image_path: Path string (for error messages).

    Returns:
        PIL.Image in RGB mode with the foci channel replicated across R, G, B.
    """
    if foci_channel < 0:
        # Auto mode: convert to RGB, but normalize if high bit-depth
        arr = np.array(raw_image)
        if arr.dtype != np.uint8:
            print(f"[classify_foci] Auto mode: {arr.dtype} image, normalizing to uint8",
                  file=sys.stderr)
            arr_8bit = normalize_to_uint8(arr)
            grayscale = Image.fromarray(arr_8bit, mode="L")
            return Image.merge("RGB", [grayscale, grayscale, grayscale])
        return raw_image.convert("RGB")

    # Strategy 1: Multi-frame TIFF (each frame = one channel)
    try:
        n_frames = getattr(raw_image, 'n_frames', 1)
        if n_frames > 1 and foci_channel < n_frames:
            raw_image.seek(foci_channel)
            channel_data = np.array(raw_image)
            print(f"[classify_foci] Extracted frame {foci_channel}/{n_frames}: "
                  f"dtype={channel_data.dtype}, min={channel_data.min()}, "
                  f"max={channel_data.max()}, mean={channel_data.mean():.1f}",
                  file=sys.stderr)
            arr_8bit = normalize_to_uint8(channel_data)
            grayscale = Image.fromarray(arr_8bit, mode="L")
            return Image.merge("RGB", [grayscale, grayscale, grayscale])
    except (EOFError, Exception) as e:
        print(f"[classify_foci] Multi-frame extraction failed: {e}", file=sys.stderr)

    # Strategy 2: Multi-band image (e.g. image.mode has multiple bands)
    try:
        bands = raw_image.split()
        if foci_channel < len(bands):
            channel_data = np.array(bands[foci_channel])
            arr_8bit = normalize_to_uint8(channel_data)
            grayscale = Image.fromarray(arr_8bit, mode="L")
            return Image.merge("RGB", [grayscale, grayscale, grayscale])
    except Exception:
        pass

    # Strategy 3: Load as numpy and index directly
    try:
        img_array = np.array(raw_image)
        if img_array.ndim == 3 and foci_channel < img_array.shape[-1]:
            channel_data = img_array[:, :, foci_channel]
            arr_8bit = normalize_to_uint8(channel_data)
            grayscale = Image.fromarray(arr_8bit, mode="L")
            return Image.merge("RGB", [grayscale, grayscale, grayscale])
    except Exception:
        pass

    # Fallback: use tifffile if available (handles complex multi-page TIFFs)
    try:
        import tifffile
        tif_data = tifffile.imread(image_path)
        # tif_data shape could be (C, H, W) or (H, W, C) or (frames, H, W)
        if tif_data.ndim == 3 and foci_channel < tif_data.shape[0]:
            channel_data = tif_data[foci_channel]
            arr_8bit = normalize_to_uint8(channel_data)
            grayscale = Image.fromarray(arr_8bit, mode="L")
            return Image.merge("RGB", [grayscale, grayscale, grayscale])
    except ImportError:
        pass
    except Exception:
        pass

    # If nothing worked, fall back to default RGB conversion
    print(f"Warning: Could not extract channel {foci_channel}, "
          f"falling back to default RGB conversion", file=sys.stderr)
    return raw_image.convert("RGB")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINO Foci Classifier")
    parser.add_argument("--image_path", required=True, help="Path to the original image")
    parser.add_argument("--mask_path", required=True, help="Path to the segmentation mask (label image)")
    parser.add_argument("--model_path", required=True, help="Path to the classification model checkpoint (.pth)")
    parser.add_argument("--backbone_name", default="vit_tiny_patch16_384",
                        help="timm backbone name")
    parser.add_argument("--image_size", type=int, default=384,
                        help="Input image size for the model")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (>= threshold => Damaged)")
    parser.add_argument("--foci_channel", type=int, default=-1,
                        help="Channel index containing foci (0-based). "
                             "-1 = auto (use all channels / default RGB conversion)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Load model
    try:
        model = FociViT(args.backbone_name, args.image_size).to(device)
        state_dict = torch.load(args.model_path, map_location=device, weights_only=False)

        # Diagnose key matching
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        matched = model_keys & ckpt_keys
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        print(f"[classify_foci] Checkpoint keys: {len(ckpt_keys)}, "
              f"Model keys: {len(model_keys)}, "
              f"Matched: {len(matched)}, "
              f"Missing from ckpt: {len(missing)}, "
              f"Unexpected in ckpt: {len(unexpected)}", file=sys.stderr)

        if len(matched) == 0:
            # Keys don't match at all — try common remappings
            # Case 1: checkpoint saved with a wrapping module (e.g. "module.xxx")
            # Case 2: checkpoint is backbone-only with different prefix
            print(f"[classify_foci] WARNING: Zero key matches! "
                  f"Sample ckpt keys: {list(ckpt_keys)[:5]}", file=sys.stderr)
            print(f"[classify_foci] Sample model keys: {list(model_keys)[:5]}", file=sys.stderr)
        elif len(missing) > 0:
            print(f"[classify_foci] Missing keys (not in ckpt): "
                  f"{sorted(missing)[:10]}", file=sys.stderr)

        result = model.load_state_dict(state_dict, strict=False)
        print(f"[classify_foci] load_state_dict result — "
              f"missing: {len(result.missing_keys)}, "
              f"unexpected: {len(result.unexpected_keys)}", file=sys.stderr)
        if result.missing_keys:
            print(f"[classify_foci] Missing keys: {result.missing_keys[:10]}", file=sys.stderr)
        if result.unexpected_keys:
            print(f"[classify_foci] Unexpected keys: {result.unexpected_keys[:10]}", file=sys.stderr)

        model.eval()
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

    # Load image — extract specific foci channel if requested
    try:
        raw_image = Image.open(args.image_path)
        image = extract_foci_channel(raw_image, args.foci_channel, args.image_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load image: {str(e)}"}))
        sys.exit(1)

    # Load mask
    try:
        mask_img = Image.open(args.mask_path)
        mask_array = np.array(mask_img)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
    except Exception as e:
        print(json.dumps({"error": f"Failed to load mask: {str(e)}"}))
        sys.exit(1)

    # Extract cell patches
    patches = extract_cell_patches(image, mask_array)
    if not patches:
        print(json.dumps({"cells": [], "summary": {"total": 0, "healthy": 0, "damaged": 0}}))
        return

    # Classify each cell
    transform = get_eval_transform(args.image_size)
    results = []

    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = 32
        for batch_start in range(0, len(patches), batch_size):
            batch_patches = patches[batch_start:batch_start + batch_size]

            tensors = []
            for label, patch_img, bbox in batch_patches:
                tensor = transform(patch_img)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors).to(device)
            probs, attn_maps = model(batch_tensor)

            for i, (label, patch_img, bbox) in enumerate(batch_patches):
                prob = probs[i].item()
                prediction = "Damaged" if prob >= args.threshold else "Healthy"

                results.append({
                    "label": label,
                    "prediction": prediction,
                    "probability": round(prob, 4),
                    "bbox": bbox
                })

    # Summary
    healthy_count = sum(1 for r in results if r["prediction"] == "Healthy")
    damaged_count = sum(1 for r in results if r["prediction"] == "Damaged")

    output = {
        "cells": results,
        "summary": {
            "total": len(results),
            "healthy": healthy_count,
            "damaged": damaged_count
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()

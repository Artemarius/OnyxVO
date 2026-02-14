#!/usr/bin/env python3
"""
Export XFeat feature extractor to ONNX format for OnyxVO.

The backbone-only export is the default and recommended path: it exports
XFeatModel (the backbone network) with fixed output shapes. NMS and top-k
selection happen in C++ on-device.

End-to-end export (full detectAndCompute) is attempted first but may fail
due to dynamic ops (nonzero in NMS). The script auto-falls back to
backbone-only.

Input: single-channel grayscale [1, 1, H, W] normalized to [0, 1].
The XFeatModel internally averages channels (identity for 1-channel) and
applies instance normalization.

Usage:
    python export_xfeat_onnx.py                          # auto (try e2e, fallback backbone)
    python export_xfeat_onnx.py --backbone-only           # backbone only
    python export_xfeat_onnx.py --verify                  # export + verify vs PyTorch

Output:
    app/src/main/assets/xfeat_fp32.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_xfeat_model():
    """Load official XFeat model via torch.hub."""
    model = torch.hub.load(
        "verlab/accelerated_features",
        "XFeat",
        pretrained=True,
        top_k=4096,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Backbone-only wrapper
# ---------------------------------------------------------------------------

class XFeatBackboneOnly(nn.Module):
    """
    Exports the XFeatModel backbone directly.

    Accepts [1, 1, H, W] grayscale input (since XFeatModel.forward() does
    x.mean(dim=1, keepdim=True) internally, 1-channel is a no-op).

    For 640x480 input, outputs at 1/8 resolution (80x60):
        feats:     [1, 64, 60, 80]  — dense local descriptors
        keypoints: [1, 65, 60, 80]  — keypoint logit map (64 spatial bins + 1 dustbin)
        heatmap:   [1,  1, 60, 80]  — reliability/confidence map
    """

    def __init__(self, xfeat_model):
        super().__init__()
        # xfeat_model is the top-level XFeat; its backbone is xfeat_model.net (XFeatModel)
        self.backbone = xfeat_model.net

    def forward(self, grayscale_image):
        # XFeatModel.forward() accepts any channel count — it does .mean(dim=1)
        feats, keypoints, heatmap = self.backbone(grayscale_image)
        return feats, keypoints, heatmap


# ---------------------------------------------------------------------------
# End-to-end wrapper (may fail due to dynamic ops)
# ---------------------------------------------------------------------------

class XFeatEndToEnd(nn.Module):
    """
    Wraps XFeat to accept [1, 1, H, W] grayscale and run full
    detectAndCompute. Returns keypoints [N, 2], descriptors [N, 64],
    scores [N].

    This may fail to export due to dynamic ops (nonzero in NMS).
    """

    def __init__(self, xfeat_model, top_k=500):
        super().__init__()
        self.xfeat = xfeat_model
        self.top_k = top_k

    def forward(self, grayscale_image):
        # XFeat's preprocess_tensor expects (B,C,H,W) — 1-channel works
        # since XFeatModel does .mean(dim=1)
        result = self.xfeat.detectAndCompute(grayscale_image, top_k=self.top_k)
        r = result[0]
        return r["keypoints"], r["descriptors"], r["scores"]


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_backbone_only(model, output_path, height=480, width=640):
    """Export backbone only (fixed output shapes)."""
    wrapper = XFeatBackboneOnly(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 1, height, width)

    print(f"Exporting backbone-only model (input=[1,1,{height},{width}])...")
    print(f"Output: {output_path}")

    # Trace-verify the wrapper produces expected shapes
    with torch.no_grad():
        feats, kpts, hmap = wrapper(dummy_input)
        print(f"  Traced shapes: feats={list(feats.shape)}, "
              f"keypoints={list(kpts.shape)}, heatmap={list(hmap.shape)}")

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["feats", "keypoints", "heatmap"],
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter (stable)
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"ONNX model validated. Size: {size_mb:.1f} MB")

    for inp in onnx_model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} {dims}")
    for out in onnx_model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")

    return True


def export_end_to_end(model, output_path, height=480, width=640, top_k=500):
    """Export full detectAndCompute pipeline to ONNX."""
    wrapper = XFeatEndToEnd(model, top_k=top_k)
    wrapper.eval()

    dummy_input = torch.randn(1, 1, height, width)

    print(f"Exporting end-to-end model (top_k={top_k}, input=[1,1,{height},{width}])...")
    print(f"Output: {output_path}")

    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            opset_version=17,
            input_names=["image"],
            output_names=["keypoints", "descriptors", "scores"],
            dynamic_axes={
                "keypoints": {0: "num_keypoints"},
                "descriptors": {0: "num_keypoints"},
                "scores": {0: "num_keypoints"},
            },
            do_constant_folding=True,
            dynamo=False,  # Use legacy exporter (stable)
        )
    except Exception as e:
        print(f"End-to-end export FAILED: {e}")
        print("This is expected — NMS uses dynamic ops (nonzero).")
        return False

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"ONNX model validated. Size: {size_mb:.1f} MB")

    for inp in onnx_model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} {dims}")
    for out in onnx_model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")

    return True


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_backbone_model(onnx_path, pytorch_model, height=480, width=640):
    """Compare backbone ONNX output vs PyTorch for a synthetic input."""
    print("\nVerifying backbone ONNX model against PyTorch...")

    test_input = np.linspace(0, 1, height * width, dtype=np.float32).reshape(1, 1, height, width)

    # --- ONNX Runtime ---
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_results = sess.run(None, {"image": test_input})

    ort_feats = ort_results[0]
    ort_kpts = ort_results[1]
    ort_hmap = ort_results[2]
    print(f"  ONNX feats: {ort_feats.shape}, keypoints: {ort_kpts.shape}, heatmap: {ort_hmap.shape}")

    # --- PyTorch ---
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_feats, pt_kpts, pt_hmap = pytorch_model.net(pt_input)
        pt_feats = pt_feats.numpy()
        pt_kpts = pt_kpts.numpy()
        pt_hmap = pt_hmap.numpy()
    print(f"  PT   feats: {pt_feats.shape}, keypoints: {pt_kpts.shape}, heatmap: {pt_hmap.shape}")

    # Compare
    feats_diff = np.abs(ort_feats - pt_feats).max()
    kpts_diff = np.abs(ort_kpts - pt_kpts).max()
    hmap_diff = np.abs(ort_hmap - pt_hmap).max()
    print(f"  Max diff — feats: {feats_diff:.6f}, keypoints: {kpts_diff:.6f}, heatmap: {hmap_diff:.6f}")

    if feats_diff < 1e-4 and kpts_diff < 1e-4 and hmap_diff < 1e-4:
        print("  VERIFICATION PASSED (diffs < 1e-4)")
    elif feats_diff < 1e-2 and kpts_diff < 1e-2 and hmap_diff < 1e-2:
        print("  VERIFICATION PASSED (diffs < 1e-2, acceptable for float32)")
    else:
        print("  WARNING: Large differences detected")

    return True


def verify_e2e_model(onnx_path, pytorch_model, height=480, width=640, top_k=500):
    """Compare end-to-end ONNX output vs PyTorch."""
    print("\nVerifying end-to-end ONNX model against PyTorch...")

    test_input = np.linspace(0, 1, height * width, dtype=np.float32).reshape(1, 1, height, width)

    # --- ONNX Runtime ---
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_results = sess.run(None, {"image": test_input})
    ort_kps = ort_results[0]
    ort_descs = ort_results[1]
    ort_scores = ort_results[2]
    print(f"  ONNX keypoints: {ort_kps.shape}, descriptors: {ort_descs.shape}, scores: {ort_scores.shape}")

    # --- PyTorch ---
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_results = pytorch_model.detectAndCompute(pt_input, top_k=top_k)
    if pt_results:
        r = pt_results[0]
        n_pt = r["keypoints"].shape[0]
        n_ort = ort_kps.shape[0]
        print(f"  Keypoint count: ONNX={n_ort}, PyTorch={n_pt}")
        pct = abs(n_ort - n_pt) / max(n_pt, 1) * 100
        if pct < 10:
            print(f"  VERIFICATION PASSED (count diff: {pct:.1f}%)")
        else:
            print(f"  WARNING: count diff: {pct:.1f}%")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export XFeat to ONNX for OnyxVO")
    parser.add_argument("--backbone-only", action="store_true",
                        help="Export backbone only (fixed output shapes)")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Max keypoints for end-to-end export (default: 500)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output matches PyTorch")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: app/src/main/assets/)")
    args = parser.parse_args()

    # Resolve output directory
    project_root = Path(__file__).resolve().parent.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "app" / "src" / "main" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "xfeat_fp32.onnx"

    # Load model
    print("Loading XFeat model...")
    model = load_xfeat_model()
    print("Model loaded successfully.")

    # Export
    is_backbone = args.backbone_only
    if args.backbone_only:
        success = export_backbone_only(model, output_path, args.height, args.width)
    else:
        success = export_end_to_end(model, output_path, args.height, args.width,
                                    top_k=args.top_k)
        if not success:
            print("\nFalling back to backbone-only export...")
            is_backbone = True
            success = export_backbone_only(model, output_path, args.height, args.width)

    if not success:
        print("Export failed!")
        sys.exit(1)

    # Verify
    if args.verify:
        if is_backbone:
            verify_backbone_model(output_path, model, args.height, args.width)
        else:
            verify_e2e_model(output_path, model, args.height, args.width, args.top_k)

    print(f"\nDone. Model saved to: {output_path}")
    print(f"Next: run quantize_xfeat.py to create INT8 model.")


if __name__ == "__main__":
    main()

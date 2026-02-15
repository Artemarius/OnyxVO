#!/usr/bin/env python3
"""
Pre-bake ONNX Runtime graph optimizations (operator fusion, constant folding)
so they don't have to run at session creation time on-device.

Usage:
    python optimize_onnx.py                      # optimize both models
    python optimize_onnx.py --verify              # verify outputs match
    python optimize_onnx.py --to-ort              # also convert to .ort format
    python optimize_onnx.py --input model.onnx    # optimize a specific model
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ASSETS_DIR = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets"

MODELS = [
    ("xfeat_fp32.onnx", "xfeat_fp32_opt.onnx"),
    ("xfeat_int8.onnx", "xfeat_int8_opt.onnx"),
]


def get_model_input_shape(session: ort.InferenceSession) -> list:
    """Extract input shape from session, replacing dynamic dims with defaults."""
    inp = session.get_inputs()[0]
    shape = []
    for dim in inp.shape:
        if isinstance(dim, int):
            shape.append(dim)
        else:
            # Dynamic dim — use 480 for H, 640 for W, 1 for batch/channel
            shape.append(1 if len(shape) < 2 else (480 if len(shape) == 2 else 640))
    return shape


def optimize_model(input_path: Path, output_path: Path) -> bool:
    """Run ORT graph optimizations and save the optimized model."""
    if not input_path.exists():
        print(f"  SKIP: {input_path.name} not found")
        return False

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(output_path)

    # Creating the session triggers optimization and saves to optimized_model_filepath
    ort.InferenceSession(str(input_path), opts, providers=["CPUExecutionProvider"])

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    delta_pct = (output_size - input_size) / input_size * 100

    print(f"  {input_path.name}: {input_size / 1024:.0f} KB -> {output_path.name}: {output_size / 1024:.0f} KB ({delta_pct:+.1f}%)")
    return True


def verify_model(original_path: Path, optimized_path: Path) -> bool:
    """Compare outputs of original vs optimized model on synthetic input."""
    if not original_path.exists() or not optimized_path.exists():
        print(f"  SKIP verify: missing model file")
        return False

    sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    sess_opt = ort.InferenceSession(str(optimized_path), providers=["CPUExecutionProvider"])

    shape = get_model_input_shape(sess_orig)
    input_name = sess_orig.get_inputs()[0].name

    # Synthetic grayscale image (uniform random, simulating normalized pixel values)
    np.random.seed(42)
    dummy_input = np.random.rand(*shape).astype(np.float32)

    outputs_orig = sess_orig.run(None, {input_name: dummy_input})
    outputs_opt = sess_opt.run(None, {input_name: dummy_input})

    all_close = True
    for i, (o_orig, o_opt) in enumerate(zip(outputs_orig, outputs_opt)):
        out_name = sess_orig.get_outputs()[i].name
        if np.allclose(o_orig, o_opt, atol=1e-5, rtol=1e-4):
            print(f"    output[{i}] ({out_name}): MATCH (shape={o_orig.shape})")
        else:
            max_diff = np.max(np.abs(o_orig - o_opt))
            print(f"    output[{i}] ({out_name}): MISMATCH (max_diff={max_diff:.6f}, shape={o_orig.shape})")
            all_close = False

    return all_close


def convert_to_ort(input_path: Path, output_path: Path) -> bool:
    """Convert ONNX model to ORT format (smaller, faster load)."""
    try:
        from onnxruntime.tools import convert_onnx_models_to_ort
    except ImportError:
        print("  SKIP .ort conversion: onnxruntime.tools not available")
        print("  Install with: pip install onnxruntime-tools")
        return False

    if not input_path.exists():
        return False

    # convert_onnx_models_to_ort works on directories
    # We use the session-based approach instead
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(output_path)
    opts.add_session_config_entry("session.save_model_format", "ORT")

    try:
        ort.InferenceSession(str(input_path), opts, providers=["CPUExecutionProvider"])
        if output_path.exists():
            print(f"  {input_path.name} -> {output_path.name} ({output_path.stat().st_size / 1024:.0f} KB)")
            return True
        else:
            print(f"  WARN: .ort output not created (may require newer ORT version)")
            return False
    except Exception as e:
        print(f"  FAIL .ort conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX models for OnyxVO")
    parser.add_argument("--input", type=str, help="Specific .onnx file to optimize")
    parser.add_argument("--verify", action="store_true", help="Verify optimized outputs match original")
    parser.add_argument("--to-ort", action="store_true", help="Also convert to .ort format")
    parser.add_argument("--assets-dir", type=str, default=str(ASSETS_DIR), help="Assets directory")
    args = parser.parse_args()

    assets = Path(args.assets_dir)
    if not assets.exists():
        print(f"ERROR: assets directory not found: {assets}")
        sys.exit(1)

    # Determine which models to process
    if args.input:
        input_path = Path(args.input)
        stem = input_path.stem
        models = [(input_path.name, f"{stem}_opt.onnx")]
        if not input_path.is_absolute():
            models = [(str(input_path), f"{stem}_opt.onnx")]
    else:
        models = MODELS

    print(f"OnyxVO ONNX Optimizer")
    print(f"Assets: {assets}")
    print(f"ORT version: {ort.__version__}")
    print()

    # Optimize
    print("Optimizing models:")
    optimized = []
    for orig_name, opt_name in models:
        orig = assets / orig_name
        opt = assets / opt_name
        if optimize_model(orig, opt):
            optimized.append((orig, opt))

    if not optimized:
        print("\nNo models found to optimize.")
        sys.exit(1)

    # Verify
    if args.verify:
        print("\nVerifying outputs:")
        all_ok = True
        for orig, opt in optimized:
            print(f"  {orig.name} vs {opt.name}:")
            if not verify_model(orig, opt):
                all_ok = False
        if all_ok:
            print("\n  All outputs match.")
        else:
            print("\n  WARNING: Some outputs differ!")
            sys.exit(1)

    # Convert to .ort
    if args.to_ort:
        print("\nConverting to .ort format:")
        for orig, opt in optimized:
            ort_path = opt.with_suffix(".ort")
            convert_to_ort(opt, ort_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

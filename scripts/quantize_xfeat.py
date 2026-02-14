#!/usr/bin/env python3
"""
Static INT8 quantization of XFeat ONNX model for OnyxVO.

Uses ONNX Runtime quantization with a calibration dataset of grayscale
images to produce a QDQ-format INT8 model optimized for mobile inference.

Usage:
    python quantize_xfeat.py                                   # use synthetic calibration
    python quantize_xfeat.py --calibration-dir ./calib_images   # use real images
    python quantize_xfeat.py --verify                           # compare FP32 vs INT8

Expects:
    app/src/main/assets/xfeat_fp32.onnx (from export_xfeat_onnx.py)

Output:
    app/src/main/assets/xfeat_int8.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

class XFeatCalibrationReader(CalibrationDataReader):
    """
    Provides calibration data for INT8 quantization.
    Uses real images from a directory, or generates synthetic patterns.
    """

    def __init__(self, input_name, height=480, width=640,
                 calibration_dir=None, num_synthetic=50):
        self.input_name = input_name
        self.height = height
        self.width = width
        self.samples = []
        self.idx = 0

        if calibration_dir and Path(calibration_dir).exists():
            self._load_real_images(calibration_dir)
        else:
            self._generate_synthetic(num_synthetic)

        print(f"Calibration dataset: {len(self.samples)} samples")

    def _load_real_images(self, cal_dir):
        """Load grayscale images from directory."""
        if Image is None:
            print("Pillow not installed, falling back to synthetic data")
            self._generate_synthetic(50)
            return

        cal_path = Path(cal_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = sorted(
            f for f in cal_path.iterdir()
            if f.suffix.lower() in extensions
        )

        if len(image_files) == 0:
            print(f"No images found in {cal_dir}, using synthetic data")
            self._generate_synthetic(50)
            return

        for img_path in image_files[:100]:  # cap at 100
            img = Image.open(img_path).convert("L")  # grayscale
            img = img.resize((self.width, self.height), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.reshape(1, 1, self.height, self.width)
            self.samples.append(arr)

        print(f"Loaded {len(self.samples)} real calibration images from {cal_dir}")

    def _generate_synthetic(self, count):
        """Generate diverse synthetic patterns for calibration."""
        rng = np.random.RandomState(42)

        for i in range(count):
            pattern_type = i % 5
            if pattern_type == 0:
                # Random noise (most common in real images)
                arr = rng.rand(self.height, self.width).astype(np.float32)
            elif pattern_type == 1:
                # Horizontal gradient
                arr = np.linspace(0, 1, self.width, dtype=np.float32)
                arr = np.tile(arr, (self.height, 1))
            elif pattern_type == 2:
                # Vertical gradient
                arr = np.linspace(0, 1, self.height, dtype=np.float32)
                arr = np.tile(arr.reshape(-1, 1), (1, self.width))
            elif pattern_type == 3:
                # Checkerboard
                block = 32
                x = np.arange(self.width) // block
                y = np.arange(self.height) // block
                arr = ((x[None, :] + y[:, None]) % 2).astype(np.float32)
                # Add noise
                arr += rng.rand(self.height, self.width).astype(np.float32) * 0.1
                arr = np.clip(arr, 0, 1)
            else:
                # Gaussian blobs (simulate textures)
                arr = np.zeros((self.height, self.width), dtype=np.float32)
                for _ in range(20):
                    cx = rng.randint(0, self.width)
                    cy = rng.randint(0, self.height)
                    sigma = rng.uniform(10, 60)
                    yy, xx = np.ogrid[:self.height, :self.width]
                    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                    arr += blob.astype(np.float32) * rng.uniform(0.3, 1.0)
                arr = np.clip(arr / arr.max(), 0, 1)

            self.samples.append(arr.reshape(1, 1, self.height, self.width))

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        data = {self.input_name: self.samples[self.idx]}
        self.idx += 1
        return data

    def rewind(self):
        self.idx = 0


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(fp32_path, int8_path, calibration_dir=None,
                   height=480, width=640):
    """Run static INT8 quantization on the FP32 model."""

    print(f"Loading FP32 model: {fp32_path}")
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    # Create calibration reader
    reader = XFeatCalibrationReader(
        input_name, height, width,
        calibration_dir=calibration_dir,
    )

    print(f"Quantizing to INT8 (QDQ format)...")
    print(f"Output: {int8_path}")

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=False,  # per-tensor for broader compatibility
        reduce_range=False,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"FP32 size: {fp32_size:.1f} MB")
    print(f"INT8 size: {int8_size:.1f} MB")
    print(f"Compression: {fp32_size / int8_size:.1f}x")

    return True


# ---------------------------------------------------------------------------
# Validation: FP32 vs INT8
# ---------------------------------------------------------------------------

def validate_quantization(fp32_path, int8_path, height=480, width=640):
    """Compare FP32 vs INT8 keypoint counts on test images."""
    print("\nValidating INT8 vs FP32...")

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    input_name_fp32 = sess_fp32.get_inputs()[0].name
    input_name_int8 = sess_int8.get_inputs()[0].name

    outputs_fp32 = [o.name for o in sess_fp32.get_outputs()]
    outputs_int8 = [o.name for o in sess_int8.get_outputs()]

    # Check if end-to-end or backbone-only
    # End-to-end has "descriptors" output; backbone has "feats"
    is_end_to_end = "descriptors" in outputs_fp32

    # Test with several patterns
    rng = np.random.RandomState(123)
    test_images = [
        ("gradient", np.linspace(0, 1, height * width, dtype=np.float32).reshape(1, 1, height, width)),
        ("noise", rng.rand(1, 1, height, width).astype(np.float32)),
        ("checkerboard", ((np.arange(width)[None, :] // 32 + np.arange(height)[:, None] // 32) % 2).astype(np.float32).reshape(1, 1, height, width)),
    ]

    total_degradation = 0
    for name, img in test_images:
        fp32_out = sess_fp32.run(outputs_fp32, {input_name_fp32: img})
        int8_out = sess_int8.run(outputs_int8, {input_name_int8: img})

        if is_end_to_end:
            kp_idx = outputs_fp32.index("keypoints")
            n_fp32 = fp32_out[kp_idx].shape[0]
            n_int8 = int8_out[kp_idx].shape[0]
            degradation = abs(n_fp32 - n_int8) / max(n_fp32, 1) * 100
            total_degradation += degradation
            print(f"  {name}: FP32={n_fp32} kps, INT8={n_int8} kps, diff={degradation:.1f}%")
        else:
            # Compare dense output magnitudes
            diff = np.abs(fp32_out[0] - int8_out[0]).mean()
            print(f"  {name}: mean feat diff = {diff:.6f}")

    if is_end_to_end:
        avg_degradation = total_degradation / len(test_images)
        print(f"\n  Average keypoint count degradation: {avg_degradation:.1f}%")
        if avg_degradation < 10:
            print("  VALIDATION PASSED (<10% degradation)")
        else:
            print("  WARNING: >10% degradation — consider adjusting quantization")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantize XFeat ONNX to INT8")
    parser.add_argument("--calibration-dir", type=str, default=None,
                        help="Directory with calibration images (50-100 grayscale)")
    parser.add_argument("--verify", action="store_true",
                        help="Validate INT8 vs FP32 keypoint quality")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--input", type=str, default=None,
                        help="Path to FP32 model (default: app/src/main/assets/xfeat_fp32.onnx)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for INT8 model (default: app/src/main/assets/xfeat_int8.onnx)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / "app" / "src" / "main" / "assets"

    fp32_path = Path(args.input) if args.input else assets_dir / "xfeat_fp32.onnx"
    int8_path = Path(args.output) if args.output else assets_dir / "xfeat_int8.onnx"

    if not fp32_path.exists():
        print(f"FP32 model not found: {fp32_path}")
        print("Run export_xfeat_onnx.py first.")
        sys.exit(1)

    success = quantize_model(
        fp32_path, int8_path,
        calibration_dir=args.calibration_dir,
        height=args.height, width=args.width,
    )

    if not success:
        sys.exit(1)

    if args.verify:
        validate_quantization(fp32_path, int8_path, args.height, args.width)

    print(f"\nDone. INT8 model: {int8_path}")
    print("Copy both .onnx files to app/src/main/assets/ and commit via Git LFS.")


if __name__ == "__main__":
    main()

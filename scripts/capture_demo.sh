#!/usr/bin/env bash
# capture_demo.sh — Capture screenshots, screen recordings, and demo GIFs via adb
# Usage:
#   ./scripts/capture_demo.sh screenshot [name]        Take a PNG screenshot
#   ./scripts/capture_demo.sh record [seconds] [name]  Record screen video
#   ./scripts/capture_demo.sh gif <input.mp4> [name]   Convert MP4 to GIF
#   ./scripts/capture_demo.sh all                      Guided capture of all README assets

set -euo pipefail

# Prevent MSYS/Git Bash from mangling /sdcard paths to C:/Program Files/Git/sdcard
export MSYS_NO_PATHCONV=1

OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/demo"
DEVICE_TMP="/sdcard/onyxvo_capture"
PACKAGE="com.onyxvo.app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Locate adb: use ANDROID_HOME if set, otherwise PATH
if [ -n "${ANDROID_HOME:-}" ] && [ -f "$ANDROID_HOME/platform-tools/adb.exe" ]; then
    ADB="$ANDROID_HOME/platform-tools/adb.exe"
elif [ -n "${ANDROID_HOME:-}" ] && [ -f "$ANDROID_HOME/platform-tools/adb" ]; then
    ADB="$ANDROID_HOME/platform-tools/adb"
else
    ADB="adb"
fi

mkdir -p "$OUT_DIR"

check_device() {
    if ! "$ADB" get-state &>/dev/null; then
        echo "ERROR: No device connected. Connect via USB and enable USB debugging."
        exit 1
    fi
    echo "Device: $("$ADB" shell getprop ro.product.model | tr -d '\r')"
}

# Android 15+ (API 35) requires explicit display ID for screencap.
# Detect once and cache.
get_display_id() {
    local api_level
    api_level=$("$ADB" shell getprop ro.build.version.sdk | tr -d '\r')
    if [ "$api_level" -ge 35 ]; then
        local display_id
        display_id=$("$ADB" shell dumpsys SurfaceFlinger --display-id 2>/dev/null \
            | head -1 | grep -oP 'Display \K[0-9]+' || echo "")
        if [ -n "$display_id" ]; then
            echo "$display_id"
            return
        fi
    fi
    echo ""
}

check_app_running() {
    local pid
    pid=$("$ADB" shell pidof "$PACKAGE" 2>/dev/null | tr -d '\r')
    if [ -z "$pid" ]; then
        echo "WARNING: $PACKAGE is not running. Launch the app first."
        return 1
    fi
    return 0
}

do_screenshot() {
    local name="${1:-screenshot_$TIMESTAMP}"
    local device_path="$DEVICE_TMP/${name}.png"
    local local_path="$OUT_DIR/${name}.png"

    check_device
    local display_id
    display_id=$(get_display_id)

    "$ADB" shell mkdir -p "$DEVICE_TMP"
    echo "Capturing screenshot..."
    if [ -n "$display_id" ]; then
        "$ADB" shell screencap -d "$display_id" "$device_path"
    else
        "$ADB" shell screencap "$device_path"
    fi
    "$ADB" pull "$device_path" "$local_path"
    "$ADB" shell rm "$device_path"
    echo "Saved: $local_path"
}

do_record() {
    local seconds="${1:-15}"
    local name="${2:-recording_$TIMESTAMP}"
    local device_path="$DEVICE_TMP/${name}.mp4"
    local local_path="$OUT_DIR/${name}.mp4"

    if [ "$seconds" -gt 180 ]; then
        echo "ERROR: screenrecord max is 180 seconds."
        exit 1
    fi

    check_device
    "$ADB" shell mkdir -p "$DEVICE_TMP"
    echo "Recording ${seconds}s of screen... (Ctrl+C to stop early)"
    echo "Operate the app now."

    # screenrecord stops on timeout or when killed
    "$ADB" shell screenrecord --time-limit "$seconds" --size 720x1280 --bit-rate 6000000 "$device_path" || true

    echo "Pulling recording..."
    "$ADB" pull "$device_path" "$local_path"
    "$ADB" shell rm "$device_path"
    echo "Saved: $local_path"
    echo ""
    echo "Convert to GIF:  ./scripts/capture_demo.sh gif $local_path"
}

do_gif() {
    local input="$1"
    local name="${2:-demo}"
    local output="$OUT_DIR/${name}.gif"

    if [ ! -f "$input" ]; then
        echo "ERROR: Input file not found: $input"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        echo "ERROR: ffmpeg not found. Install it to convert MP4 to GIF."
        echo "  Windows: winget install ffmpeg"
        echo "  macOS:   brew install ffmpeg"
        echo "  Linux:   sudo apt install ffmpeg"
        exit 1
    fi

    echo "Converting to GIF (360px wide, 8 fps, 10s max)..."
    ffmpeg -y -i "$input" \
        -t 10 \
        -vf "fps=8,scale=360:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
        "$output"

    local size
    size=$(du -h "$output" | cut -f1)
    echo "Saved: $output ($size)"
    if [ "$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)" -gt 10485760 ]; then
        echo "WARNING: GIF is >10 MB. GitHub READMEs load slowly above this. Consider trimming."
    fi
}

do_all() {
    check_device
    echo "=== OnyxVO Demo Asset Capture ==="
    echo "Make sure the app is running with a good scene visible."
    echo ""

    if ! check_app_running; then
        echo "Starting $PACKAGE..."
        "$ADB" shell am start -n "$PACKAGE/.MainActivity"
        sleep 3
    fi

    # Screenshot 1: Keypoint overlay
    echo ""
    echo "--- Screenshot 1: Keypoint Overlay ---"
    echo "Point the camera at a textured scene so keypoints are visible."
    read -rp "Press Enter when ready..."
    do_screenshot "keypoint_overlay"

    # Screenshot 2: Trajectory view
    echo ""
    echo "--- Screenshot 2: Trajectory View ---"
    echo "Move the phone to build a trajectory, then hold steady."
    read -rp "Press Enter when ready..."
    do_screenshot "trajectory_view"

    # Screenshot 3: Performance dashboard
    echo ""
    echo "--- Screenshot 3: Performance Dashboard ---"
    echo "Make sure the performance dashboard is visible on screen."
    read -rp "Press Enter when ready..."
    do_screenshot "performance_dashboard"

    # Screen recording for demo GIF
    echo ""
    echo "--- Demo Recording (15s) ---"
    echo "Walk around slowly to show trajectory building in real-time."
    read -rp "Press Enter to start recording..."
    do_record 15 "demo_recording"

    # Convert to GIF
    echo ""
    echo "--- Converting to GIF ---"
    do_gif "$OUT_DIR/demo_recording.mp4" "demo"

    echo ""
    echo "=== All assets captured ==="
    echo "Files in: $OUT_DIR/"
    ls -lh "$OUT_DIR/"
    echo ""
    echo "Assets are already referenced in README.md."
}

usage() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  screenshot [name]            Take a PNG screenshot"
    echo "  record [seconds] [name]      Record screen (default: 15s, max: 180s)"
    echo "  gif <input.mp4> [name]       Convert MP4 to GIF (360px, 8fps, 10s)"
    echo "  all                          Guided capture of all README assets"
    echo ""
    echo "Examples:"
    echo "  $0 screenshot keypoint_overlay"
    echo "  $0 record 15 demo_walk"
    echo "  $0 gif assets/demo/demo_walk.mp4 demo"
    echo "  $0 all"
}

case "${1:-}" in
    screenshot) do_screenshot "${2:-}" ;;
    record)     do_record "${2:-15}" "${3:-}" ;;
    gif)        do_gif "${2:?Missing input MP4}" "${3:-demo}" ;;
    all)        do_all ;;
    *)          usage ;;
esac

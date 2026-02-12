#!/usr/bin/env bash
set -e

LOG_DIR="/advantech/diagnostics"
LOG_FILE="${LOG_DIR}/wise-bench.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# =========================
# UI helpers
# =========================
print_header() {
    echo
    echo "▶ $1"
    echo
}

print_table_header() {
    echo "+------------------------+--------------------------------+"
    printf "| %-22s | %-30s |\n" "$1" ""
    echo "+------------------------+--------------------------------+"
}

print_table_row() {
    printf "| %-22s | %-30s |\n" "$1" "$2"
}

print_table_footer() {
    echo "+------------------------+--------------------------------+"
}

# =========================
# Banner
# =========================
clear
echo "+------------------------------------------------------+"
echo "|    Advantech YOLO Vision Hardware Diagnostics Tool   |"
echo "+------------------------------------------------------+"
echo
cat <<'EOF'
       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗
      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║
      ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║
      ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║
      ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║
      ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝
                           Center of Excellence
EOF

echo
echo "▶ Starting AMD Ryzen AI diagnostics..."
echo "  This may take a moment..."
sleep 1


# =========================
# Python Info
# =========================
print_header "PYTHON ENVIRONMENT"

PY_VERSION=$(python3 --version 2>/dev/null || echo "Not Found")
print_table_header "PYTHON"
print_table_row "Python Version" "$PY_VERSION"
print_table_footer

# =========================
# Ryzen AI NPU Driver (XRT)
# =========================
print_header "RYZEN AI NPU DRIVER (XRT)"

XRT_OK=0
NPU_NAME="Unknown"

if command -v xrt-smi >/dev/null 2>&1; then
    if xrt-smi examine >/dev/null 2>&1; then
        XRT_OK=1

        NPU_NAME=$(xrt-smi examine 2>/dev/null | \
            awk -F'|' '
            /^\|/ && $2 ~ /\[/ {
                gsub(/^ +| +$/, "", $3)
                print $3
                exit
            }')

        [[ -z "$NPU_NAME" ]] && NPU_NAME="Unknown NPU"
    fi
fi


print_table_header "XRT STATUS"

if [[ $XRT_OK -eq 1 ]]; then
    print_table_row "AMD Ryzen AI NPU" "✓ Available"
    print_table_row "NPU Name" "$NPU_NAME"
else
    print_table_row "AMD Ryzen AI NPU" "✗ Not detected"
fi

print_table_footer

# =========================
# OpenCV
# =========================
print_header "OPENCV TEST"

OPENCV_OK=0
OPENCV_VERSION=$(python3 - <<EOF 2>/dev/null || true
import cv2
print(cv2.__version__)
EOF
)

if [[ -n "$OPENCV_VERSION" ]]; then
    OPENCV_OK=1
else
    OPENCV_VERSION="Not installed"
fi

print_table_header "OPENCV DETAILS"
print_table_row "OpenCV Version" "$OPENCV_VERSION"
[[ $OPENCV_OK -eq 1 ]] \
    && print_table_row "Status" "✓ Available" \
    || print_table_row "Status" "✗ Not installed"
print_table_footer

# =========================
# PyTorch
# =========================
print_header "PYTORCH TEST"

PYTORCH_OK=0
PYTORCH_VERSION=$(python3 - <<EOF 2>/dev/null || true
import torch
print(torch.__version__)
EOF
)

if [[ -n "$PYTORCH_VERSION" ]]; then
    PYTORCH_OK=1
else
    PYTORCH_VERSION="Not installed"
fi

print_table_header "PYTORCH DETAILS"
print_table_row "PyTorch Version" "$PYTORCH_VERSION"
[[ $PYTORCH_OK -eq 1 ]] \
    && print_table_row "Status" "✓ Available" \
    || print_table_row "Status" "✗ Not installed"
print_table_footer

# =========================
# ONNX Runtime
# =========================
print_header "ONNX RUNTIME"

ONNX_OK=0
ONNX_NPU_OK=0

ONNX_PROVIDERS=$(python3 - <<EOF 2>/dev/null || true
import onnxruntime as ort
print("\n".join(ort.get_available_providers()))
EOF
)

if [[ -n "$ONNX_PROVIDERS" ]]; then
    ONNX_OK=1
    echo "$ONNX_PROVIDERS" | grep -q "VitisAIExecutionProvider" && ONNX_NPU_OK=1
fi

print_table_header "ONNX PROVIDERS"

if [[ $ONNX_OK -eq 1 && $ONNX_NPU_OK -eq 1 ]]; then
    print_table_row "ONNX Runtime" "✓ NPU Accelerated"
elif [[ $ONNX_OK -eq 1 ]]; then
    print_table_row "ONNX Runtime" "⚠ CPU Only"
else
    print_table_row "ONNX Runtime" "✗ Not installed"
fi

print_table_footer

# =========================
# Camera
# =========================
print_header "CAMERA (V4L2)"

CAM_OK=0
if compgen -G "/dev/video*" > /dev/null; then
    CAM_OK=1
fi

print_table_header "CAMERA"
[[ $CAM_OK -eq 1 ]] \
    && print_table_row "Camera (V4L2)" "✓ Available" \
    || print_table_row "Camera (V4L2)" "✗ Not found"
print_table_footer

# =========================
# YOLOv11 TEST
# =========================
print_header "YOLOv11 OBJECT DETECTION (ONNX)"

YOLO_MODEL="/workspace/models/yolo11n.onnx"
YOLO_IMAGE="/workspace/data/images/test.jpg"
YOLO_OK=0
YOLO_DEVICE="Unknown"
YOLO_DETAILS="N/A"
YOLO_LATENCY="N/A"

PY_SCRIPT="/tmp/yolo_test.py"

cat > "$PY_SCRIPT" << 'EOF'
import os, sys, time, cv2, numpy as np
import onnxruntime as ort

model = sys.argv[1]
image_path = sys.argv[2]

if not os.path.exists(model):
    print("ERROR=MODEL_NOT_FOUND")
    sys.exit(1)

if not os.path.exists(image_path):
    print("ERROR=IMAGE_NOT_FOUND")
    sys.exit(1)

providers = ort.get_available_providers()
if "VitisAIExecutionProvider" not in providers:
    print("ERROR=NPU_NOT_AVAILABLE")
    sys.exit(1)

session = ort.InferenceSession(model, providers=["VitisAIExecutionProvider"])
input_meta = session.get_inputs()[0]
input_shape = input_meta.shape

if len(input_shape) == 4 and isinstance(input_shape[2], int):
    size = input_shape[2]
else:
    size = 640

img = cv2.imread(image_path)
img = cv2.resize(img, (size, size))
img = img.astype(np.float32) / 255.0
img = img.transpose(2,0,1)[None,:,:,:]

# Warmup
session.run(None, {input_meta.name: img})

start = time.perf_counter()
outputs = session.run(None, {input_meta.name: img})
latency = (time.perf_counter() - start) * 1000

print("DEVICE=AMD_NPU")
print(f"LATENCY_MS={latency:.2f}")
print(f"OUTPUT_SHAPE={outputs[0].shape}")
print("STATUS=SUCCESS")
EOF

if python3 "$PY_SCRIPT" "$YOLO_MODEL" "$YOLO_IMAGE" > /tmp/yolo.log 2>/dev/null; then
    YOLO_OK=1
    YOLO_DEVICE=$(grep DEVICE /tmp/yolo.log | cut -d= -f2- || true)
    YOLO_LATENCY=$(grep LATENCY_MS /tmp/yolo.log | cut -d= -f2- || true)
    YOLO_DETAILS="Inference OK"
else
    YOLO_DETAILS=$(grep ERROR /tmp/yolo.log | cut -d= -f2- || echo "Test failed")
fi

print_table_header "YOLOv11 DETAILS"
print_table_row "Model" "$(basename "$YOLO_MODEL")"
print_table_row "Device" "$YOLO_DEVICE"

if [[ $YOLO_OK -eq 1 ]]; then
    print_table_row "Status" "✓ Working"
    print_table_row "Latency (ms)" "$YOLO_LATENCY"
else
    print_table_row "Status" "✗ Failed"
    print_table_row "Details" "$YOLO_DETAILS"
fi

print_table_footer

# =========================
# FINAL SUMMARY
# =========================
print_header "FINAL DIAGNOSTIC SUMMARY"

print_table_header "SYSTEM STATUS"

[[ $XRT_OK -eq 1 ]] \
    && print_table_row "AMD Ryzen AI NPU" "✓ Available" \
    || print_table_row "AMD Ryzen AI NPU" "✗ Not detected"

if [[ $ONNX_OK -eq 1 && $ONNX_NPU_OK -eq 1 ]]; then
    print_table_row "ONNX Runtime" "✓ NPU Accelerated"
elif [[ $ONNX_OK -eq 1 ]]; then
    print_table_row "ONNX Runtime" "⚠ CPU Only"
else
    print_table_row "ONNX Runtime" "✗ Not installed"
fi

[[ $OPENCV_OK -eq 1 ]] \
    && print_table_row "OpenCV" "✓ Available" \
    || print_table_row "OpenCV" "✗ Missing"

[[ $PYTORCH_OK -eq 1 ]] \
    && print_table_row "PyTorch" "✓ Available" \
    || print_table_row "PyTorch" "✗ Missing"

[[ $CAM_OK -eq 1 ]] \
    && print_table_row "Camera (V4L2)" "✓ Available" \
    || print_table_row "Camera (V4L2)" "✗ Not found"

[[ $YOLO_OK -eq 1 ]] \
    && print_table_row "YOLOv11 (ONNX)" "✓ Working" \
    || print_table_row "YOLOv11 (ONNX)" "✗ Failed"

print_table_footer

echo
echo "▶ Diagnostics completed."
echo "Log file saved to: $LOG_FILE"


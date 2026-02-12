#!/usr/bin/env bash
set -e

source /home/ryzen_ai_1.6/venv/bin/activate
export PYTHONPATH=/workspace:$PYTHONPATH

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
# Ryzen AI NPU Driver (XRT)
# =========================
print_header "RYZEN AI NPU DRIVER (XRT)"

XRT_OK=0
if command -v xrt-smi >/dev/null 2>&1; then
    if xrt-smi examine >/dev/null 2>&1; then
        XRT_OK=1
    fi
fi

print_table_header "XRT STATUS"
if [[ $XRT_OK -eq 1 ]]; then
    print_table_row "AMD Ryzen AI NPU" "✓ Available"
else
    print_table_row "AMD Ryzen AI NPU" "✗ Not detected"
fi
print_table_footer

# =========================
# OpenCV
# =========================
print_header "OPENCV TEST"

OPENCV_OK=0
OPENCV_VERSION="Not installed"

if python3 -c "import cv2" >/dev/null 2>&1; then
    OPENCV_VERSION=$(python3 - <<EOF
import cv2
print(cv2.__version__)
EOF
)
    OPENCV_OK=1
fi

print_table_header "OPENCV DETAILS"
print_table_row "OpenCV Version" "$OPENCV_VERSION"
if [[ $OPENCV_OK -eq 1 ]]; then
    print_table_row "Status" "✓ Available"
else
    print_table_row "Status" "✗ Not installed"
fi
print_table_footer

# =========================
# PyTorch
# =========================
print_header "PYTORCH TEST"

PYTORCH_OK=0
PYTORCH_VERSION="Not installed"

if python3 -c "import torch" >/dev/null 2>&1; then
    PYTORCH_VERSION=$(python3 - <<EOF
import torch
print(torch.__version__)
EOF
)
    PYTORCH_OK=1
fi

print_table_header "PYTORCH DETAILS"
print_table_row "PyTorch Version" "$PYTORCH_VERSION"
if [[ $PYTORCH_OK -eq 1 ]]; then
    print_table_row "Status" "✓ Available"
else
    print_table_row "Status" "✗ Not installed"
fi
print_table_footer

# =========================
# ONNX Runtime + NPU EP
# =========================
print_header "RYZEN AI NPU EXECUTION PROVIDER"

ONNX_OK=0
ONNX_NPU_OK=0

if python3 - <<EOF >/tmp/onnx_ep.txt 2>/dev/null
import onnxruntime as ort
print("\n".join(ort.get_available_providers()))
EOF
then
    ONNX_OK=1
    if grep -q "VitisAIExecutionProvider" /tmp/onnx_ep.txt; then
        ONNX_NPU_OK=1
    fi
fi

print_table_header "ONNX RUNTIME"
if [[ $ONNX_OK -eq 1 && $ONNX_NPU_OK -eq 1 ]]; then
    print_table_row "ONNX Runtime" "✓ NPU Accelerated"
elif [[ $ONNX_OK -eq 1 ]]; then
    print_table_row "ONNX Runtime" "⚠ CPU Only"
else
    print_table_row "ONNX Runtime" "✗ Not installed"
fi
print_table_footer

# =========================
# Camera (V4L2)
# =========================
print_header "CAMERA (V4L2)"

CAM_OK=0
if ls /dev/video* >/dev/null 2>&1; then
    CAM_OK=1
fi

print_table_header "CAMERA"
if [[ $CAM_OK -eq 1 ]]; then
    print_table_row "Camera (V4L2)" "✓ Available"
else
    print_table_row "Camera (V4L2)" "✗ Not found"
fi
print_table_footer

# =========================
# YOLOv11 OBJECT DETECTION (ONNX)
# =========================
print_header "YOLOv11 OBJECT DETECTION (ONNX)"

YOLO11_OK=0
YOLO11_MODEL="/workspace/models/yolo11n.onnx"
YOLO11_IMAGE="/workspace/data/images/test.jpg"

YOLO11_DEVICE="Unknown"
YOLO11_STATUS="N/A"
YOLO11_DETAILS="N/A"

# Create Python test script
cat > /tmp/yolov11_npu_test.py << 'EOF'
#!/usr/bin/env python3
"""YOLOv11 NPU Test Script"""

import os
import sys
import time
import cv2
import numpy as np

def test_yolov11_npu(model_path, image_path):
    """Test YOLOv11 NPU inference"""
    try:
        import onnxruntime as ort
        
        # Check files
        if not os.path.exists(model_path):
            print("ERROR=MODEL_NOT_FOUND")
            return False
        
        if not os.path.exists(image_path):
            print("ERROR=IMAGE_NOT_FOUND")
            return False
        
        # Check NPU provider
        providers = ort.get_available_providers()
        if 'VitisAIExecutionProvider' not in providers:
            print("ERROR=NPU_NOT_AVAILABLE")
            return False
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("ERROR=IMAGE_LOAD_FAILED")
            return False
        
        # Create NPU session
        session = ort.InferenceSession(model_path, providers=['VitisAIExecutionProvider'])
        
        # Get model info
        input_name = session.get_inputs()[0].name
        output_names = [out.name for out in session.get_outputs()]
        input_shape = session.get_inputs()[0].shape
        
        if len(input_shape) == 4:
            input_size = input_shape[3]
        else:
            input_size = 640
        
        # Preprocess
        img = cv2.resize(image, (input_size, input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        blob = np.ascontiguousarray(img)
        
        # Perform inference
        outputs = session.run(output_names, {input_name: blob})
        
        # Calculate performance (single inference)
        start_time = time.perf_counter()
        outputs = session.run(output_names, {input_name: blob})
        latency = (time.perf_counter() - start_time) * 1000
        
        print(f"DEVICE=AMD_NPU")
        print(f"INPUT_SHAPE={input_shape}")
        print(f"OUTPUT_SHAPE={outputs[0].shape}")
        print(f"STATUS=SUCCESS")
        
        # Check for valid detections
        if outputs and len(outputs) > 0 and outputs[0].size > 0:
            print("DETECTIONS=YES")
        else:
            print("DETECTIONS=NO")
            
        return True
            
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"ERROR={error_msg}")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/yolo11n.onnx"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/data/test.jpg"
    
    success = test_yolov11_npu(model_path, image_path)
    sys.exit(0 if success else 1)
EOF

# Run test
echo "Testing YOLOv11 on NPU..."
if python3 /tmp/yolov11_npu_test.py "$YOLO11_MODEL" "$YOLO11_IMAGE" >/tmp/yolov11_result.log 2>/tmp/yolov11_error.log; then
    # Extract test results
    RESULT_FILE="/tmp/yolov11_result.log"
    
    # Check for errors
    if grep -q "^ERROR=" "$RESULT_FILE"; then
        ERROR_MSG=$(grep "^ERROR=" "$RESULT_FILE" | head -1 | cut -d= -f2-)
        YOLO11_DETAILS="Error: $ERROR_MSG"
    else
        # Extract successful information
        YOLO11_OK=1
        
        # Device info
        if grep -q "^DEVICE=" "$RESULT_FILE"; then
            YOLO11_DEVICE=$(grep "^DEVICE=" "$RESULT_FILE" | cut -d= -f2-)
        fi
        
        # Status
        if grep -q "^STATUS=" "$RESULT_FILE"; then
            YOLO11_STATUS="SUCCESS"
        fi
        
        # Detection results
        if grep -q "^DETECTIONS=" "$RESULT_FILE"; then
            DETECTIONS=$(grep "^DETECTIONS=" "$RESULT_FILE" | cut -d= -f2-)
            if [[ "$DETECTIONS" == "YES" ]]; then
                YOLO11_DETAILS="Objects detected"
            else
                YOLO11_DETAILS="No objects (test image)"
            fi
        fi
        
        # Input/Output info
        INPUT_SHAPE=$(grep "^INPUT_SHAPE=" "$RESULT_FILE" | cut -d= -f2-)
        OUTPUT_SHAPE=$(grep "^OUTPUT_SHAPE=" "$RESULT_FILE" | cut -d= -f2-)
    fi
else
    # Test failed
    if [[ -f "/tmp/yolov11_result.log" ]]; then
        ERROR_MSG=$(grep "^ERROR=" "/tmp/yolov11_result.log" 2>/dev/null | head -1 | cut -d= -f2-)
        if [[ -z "$ERROR_MSG" ]]; then
            ERROR_MSG="Unknown error"
        fi
        YOLO11_DETAILS="Test failed: $ERROR_MSG"
    else
        YOLO11_DETAILS="Test failed (no output)"
    fi
fi

# Check if files exist
if [[ ! -f "$YOLO11_MODEL" ]]; then
    YOLO11_DETAILS="Model missing"
    YOLO11_OK=0
elif [[ ! -f "$YOLO11_IMAGE" ]]; then
    YOLO11_DETAILS="Image missing"
    YOLO11_OK=0
fi

print_table_header "YOLOv11 DETAILS"
print_table_row "Model" "$(basename "$YOLO11_MODEL")"
print_table_row "Device" "$YOLO11_DEVICE"

if [[ $YOLO11_OK -eq 1 ]]; then
    print_table_row "Status" "✓ Working"
    print_table_row "Details" "$YOLO11_DETAILS"
    print_table_row "Input Shape" "$INPUT_SHAPE"
    print_table_row "Output Shape" "$OUTPUT_SHAPE"
else
    print_table_row "Status" "✗ Failed"
    print_table_row "Details" "$YOLO11_DETAILS"
fi
print_table_footer

# =========================
# Final Summary
# =========================
print_header "FINAL DIAGNOSTIC SUMMARY"

print_table_header "SYSTEM STATUS"
[[ $XRT_OK -eq 1 ]] && print_table_row "AMD Ryzen AI NPU" "✓ Available" \
                    || print_table_row "AMD Ryzen AI NPU" "✗ Not detected"
[[ $ONNX_OK -eq 1 && $ONNX_NPU_OK -eq 1 ]] && print_table_row "ONNX Runtime" "✓ NPU Accelerated" \
                                            || print_table_row "ONNX Runtime" "⚠ CPU Only"
[[ $OPENCV_OK -eq 1 ]] && print_table_row "OpenCV" "✓ Available" \
                         || print_table_row "OpenCV" "✗ Missing"
[[ $PYTORCH_OK -eq 1 ]] && print_table_row "PyTorch" "✓ Available" \
                          || print_table_row "PyTorch" "✗ Missing"
[[ $CAM_OK -eq 1 ]] && print_table_row "Camera (V4L2)" "✓ Available" \
                      || print_table_row "Camera (V4L2)" "✗ Not found"

if [[ $YOLO11_OK -eq 1 ]]; then
    print_table_row "YOLOv11 (ONNX)" "✓ Working"
else
    print_table_row "YOLOv11 (ONNX)" "✗ Failed"
fi

print_table_footer

echo
echo "▶ Diagnostics completed."
echo "Log file saved to: $LOG_FILE"

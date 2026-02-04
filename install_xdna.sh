#!/usr/bin/env bash
set -e

############################################
# AMD XDNA / NPU Installation Script
############################################

#------------------------------------------------
# 0. WORKDIR: Current script directory
#------------------------------------------------
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/amd/xdna-driver.git"
REPO_COMMIT="ca51aa3ad20a7133eecd04ce34666de5fed51d3c"

echo "=========================================="
echo " AMD XDNA (NPU) Installation Script"
echo " Commit: $REPO_COMMIT"
echo "=========================================="

#------------------------------------------------
# 1. Check kernel version (>= 6.10)
#------------------------------------------------
KERNEL_VER=$(uname -r | cut -d- -f1)
REQUIRED_VER="6.10"

verlte() { printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1 | grep -qx "$1"; }

if verlte "$REQUIRED_VER" "$KERNEL_VER"; then
    echo "[OK] Kernel version: $KERNEL_VER"
else
    echo "[ERROR] Kernel $KERNEL_VER < 6.10"
    echo "Please upgrade kernel (HWE stack recommended)."
    exit 1
fi

#------------------------------------------------
# 2. Install build dependencies
#------------------------------------------------
echo "[INFO] Installing build dependencies..."
sudo apt update
sudo apt install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libdrm-dev \
    libpciaccess-dev \
    libboost-all-dev \
    libssl-dev \
    python3 \
    python3-pip \
    ca-certificates

#------------------------------------------------
# 3. Clone xdna-driver (pinned commit)
#------------------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d xdna-driver ]; then
    echo "[INFO] Cloning xdna-driver..."
    git clone "$REPO_URL"
fi

cd xdna-driver
echo "[INFO] Checkout commit $REPO_COMMIT"
git fetch --all
git checkout "$REPO_COMMIT"

#------------------------------------------------
# 4. Init submodules
#------------------------------------------------
echo "[INFO] Initializing submodules..."
git submodule update --init --recursive

#------------------------------------------------
# 5. Install amdxdna dependencies
#------------------------------------------------
echo "[INFO] Installing amdxdna build dependencies..."
sudo ./tools/amdxdna_deps.sh

#------------------------------------------------
# 6. Build XRT (NPU)
#------------------------------------------------
echo "[INFO] Building XRT (NPU)..."
cd xrt/build
./build.sh -npu -opt

#------------------------------------------------
# 7. Install XRT DEB or fallback tar.gz
#------------------------------------------------
cd Release
if ls ./*.deb 1> /dev/null 2>&1; then
    echo "[INFO] Installing XRT DEB packages..."
    sudo apt install -y ./*.deb
else
    echo "[INFO] DEB packages not found, using tar.gz fallback..."
    for f in ./*.tar.gz; do
        echo "[INFO] Extracting $f ..."
        sudo tar -xzvf "$f" -C /
    done
fi
cd ../../..

#------------------------------------------------
# 8. Build and install XRT plugin
#------------------------------------------------
cd build
./build.sh -release
cd Release
if ls ./xrt_plugin*-amdxdna.deb 1> /dev/null 2>&1; then
    echo "[INFO] Installing XRT plugin DEB..."
    sudo apt install -y ./xrt_plugin*-amdxdna.deb
else
    echo "[WARNING] Plugin DEB not found! Please check build log."
fi
cd ../..

#------------------------------------------------
# 9. Validation
#------------------------------------------------
echo "[INFO] Validating XDNA / NPU..."
source /opt/xilinx/xrt/setup.sh

if command -v xbutil >/dev/null 2>&1; then
    xbutil examine
else
    xrt-smi validate
fi

echo "=========================================="
echo " XDNA installation completed successfully "
echo "=========================================="


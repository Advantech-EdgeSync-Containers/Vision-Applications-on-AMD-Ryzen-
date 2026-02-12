#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# Ryzen AI / NPU Docker Pull & Setup Script (NO Dockerfile)
# ==========================================================================
NC='\033[0m'
WHITE='\033[1;37m'
CYAN='\033[1;36m'
BLUE='\033[0;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'

REGISTRY_IMAGE="harbor.edgesync.cloud/amd/ryzen_ai_npu_ubuntu24"
LOCAL_TAG="ryzen_ai_npu:latest"

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }
log_error() { echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2; }
log_warning() { echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2; }

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
display_banner() {
    clear
    echo -e "${BLUE}"
    cat << 'EOF'
       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗     ██████╗ ██████╗ ███████╗
      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║    ██╔════╝██╔═══██╗██╔════╝
      ███████║██║  ██║██║   ██║███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║    ██║     ██║   ██║█████╗  
      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║    ██║     ██║   ██║██╔══╝  
      ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║    ╚██████╗╚██████╔╝███████╗
      ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚══════╝
EOF
    echo -e "${WHITE}                                  Center of Excellence${NC}"
    echo
    echo -e "${CYAN}Initializing AI Development Environment...${NC}\n"
    sleep 2
}

# -----------------------------------------------------------------------------
# Docker environment check
# -----------------------------------------------------------------------------
check_docker() {
    log "Checking Docker environment..."

    if ! command -v docker &>/dev/null; then
        log_error "Docker not installed"
        exit 1
    fi

    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
        log_error "docker-compose not found"
        exit 1
    fi

    log_success "Docker environment verified"
}

# -----------------------------------------------------------------------------
# Install system runtime libraries
# -----------------------------------------------------------------------------
install_runtime_libs() {
    log "Installing required system runtime libraries..."
    sudo apt update
    sudo apt install -y \
        libpcre3 \
        libboost-filesystem1.74.0 \
        libboost-system1.74.0 \
        libboost-thread1.74.0
    log_success "Runtime libraries installed"
}

# -----------------------------------------------------------------------------
# Verify Boost ABI
# -----------------------------------------------------------------------------
verify_boost() {
    log "Verifying Boost 1.74 runtime..."
    for lib in filesystem system thread; do
        if [[ ! -f "/usr/lib/x86_64-linux-gnu/libboost_${lib}.so.1.74.0" ]]; then
            log_error "libboost_${lib}.so.1.74.0 not found"
            log_error "Hint: Install Ubuntu 20.04 Boost 1.74 manually"
            exit 1
        fi
    done
    log_success "Boost 1.74 runtime verified"
}

# -----------------------------------------------------------------------------
# Host runtime check (XRT)
# -----------------------------------------------------------------------------
check_host_runtime() {
    log "Checking host NPU runtime..."
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    source /opt/xilinx/xrt/setup.sh

    if ! command -v xrt-smi >/dev/null; then
        log_error "xrt-smi not found"
        exit 1
    fi

    xrt-smi examine
    log_success "Host NPU runtime OK"
}


# -----------------------------------------------------------------------------
# Pull & tag image
# -----------------------------------------------------------------------------
pull_image() {
    local src="${REGISTRY_IMAGE}:v2.0"

    log "Pulling latest image:"
    log "  ${src}"

    docker pull "${src}"

    log "Tagging image as ${LOCAL_TAG}"
    docker tag "${src}" "${LOCAL_TAG}"

    log_success "Image ready: ${LOCAL_TAG}"
}

# -----------------------------------------------------------------------------
# Launch container
# -----------------------------------------------------------------------------
start_compose() {
    log "Starting container via docker-compose..."
    docker compose up -d
    log_success "Ryzen AI NPU container is running"
}

# -----------------------------------------------------------------------------
# Enable X11 display permissions
# -----------------------------------------------------------------------------
enable_x11() {
    log "Enabling X11 display permissions..."
    if ! command -v xhost &>/dev/null; then
        log_error "xhost command not found - install x11-xserver-utils package"
        exit 1
    fi
    
    xhost +local:docker
    log_success "X11 permissions granted to Docker containers"
}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    display_banner
    check_docker
    install_runtime_libs
    verify_boost
    check_host_runtime
    enable_x11
    pull_image
    start_compose
    log_success "NPU runtime environment is ready"
}

main "$@"


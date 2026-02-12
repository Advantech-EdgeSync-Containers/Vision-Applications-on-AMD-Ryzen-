#!/usr/bin/env python3
__version__ = "2.2.0"
__author__ = "Advantech Co., Ltd"
__build_date__ = "2026-1"
__copyright__ = "Copyright (c) 2026 Advantech Co., Ltd. All Rights Reserved."

import sys
import os
import signal
import threading
import time
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# ==========================================================================
# Fix Qt font warning issues
# ==========================================================================
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'

# Set common environment variables to reduce log output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress OpenCV Qt-related warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import warnings
warnings.filterwarnings('ignore')

# ==========================================================================
# Global variables and signal handling
# ==========================================================================
WINDOW_NAME = "Advantech Yolo Vision (AMD NPU)"
_GLOBAL_SHUTDOWN = threading.Event()
_FORCE_EXIT_COUNT = 0

def force_shutdown_handler(signum, frame):
    global _FORCE_EXIT_COUNT
    _FORCE_EXIT_COUNT += 1
    _GLOBAL_SHUTDOWN.set()
    if _FORCE_EXIT_COUNT == 1:
        print("\n\n[!] Shutdown requested (Ctrl+C again to force quit)...")
    elif _FORCE_EXIT_COUNT >= 2:
        print("\n[!] Force quitting...")
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
        os._exit(1)

signal.signal(signal.SIGINT, force_shutdown_handler)
signal.signal(signal.SIGTERM, force_shutdown_handler)

def is_shutdown() -> bool:
    return _GLOBAL_SHUTDOWN.is_set()

def request_shutdown():
    _GLOBAL_SHUTDOWN.set()

# ==========================================================================
# Import Core Modules (Enhanced with Performance Comparison)
# ==========================================================================
try:
    from advantech_core import (
        AdvantechConfig, AdvantechLogger, AdvantechTaskType, AdvantechModelFormat,
        AdvantechInputSource, AdvantechDetection, AdvantechClassification,
        AdvantechEngine, AdvantechOnnxEngine, AdvantechPyTorchEngine, AdvantechModelLoader,
        AdvantechMetricsCollector, AdvantechMetrics, AdvantechOverlayRenderer,
        AdvantechMemoryManager, AdvantechRingBuffer, AdvantechFrame,
        AdvantechPerformanceMetrics, AdvantechPerformanceComparison,
        AdvantechPerformanceMonitor, AdvantechPerformanceComparer,
        detect_amd_npu_hardware, get_amd_npu_config,
        ONNX_AVAILABLE, TORCH_AVAILABLE, CV2_AVAILABLE
    )
    CORE_IMPORTED = True
    PERFORMANCE_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Error: Cannot import advantech_core module: {e}")
    print("Please ensure advantech_core.py file is in current directory")
    sys.exit(1)

# ==========================================================================
# Check for required packages
# ==========================================================================
if not CV2_AVAILABLE:
    print("Error: OpenCV not installed. Please install with: pip install opencv-python")
    sys.exit(1)

if not ONNX_AVAILABLE:
    print("Warning: ONNX Runtime not installed. ONNX models will not work.")
    print("Install with: pip install onnxruntime")

# ==========================================================================
# Set more environment variables before importing cv2
# ==========================================================================
os.environ['QT_FONT_DPI'] = '96'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'


try:
    import cv2
    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False
    print("Error: Failed to import cv2. Please install OpenCV.")
    sys.exit(1)

import numpy as np

# ==========================================================================
# Data Class Definitions
# ==========================================================================
@dataclass
class CameraFormat:
    pixel_format: str
    width: int
    height: int
    fps: List[int] = field(default_factory=list)

# ==========================================================================
# Camera Discovery Class
# ==========================================================================
class AdvantechCameraDiscovery:
    @staticmethod
    def _suppress_errors():
        os.environ['GST_DEBUG'] = '0'
        os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
    
    @staticmethod
    def _test_capture(device_num: int) -> bool:
        if is_shutdown():
            return False
        try:
            stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            
            try:
                cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(device_num)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(f'/dev/video{device_num}')
                
                if not cap.isOpened():
                    return False
                
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                for _ in range(3):
                    if is_shutdown():
                        cap.release()
                        return False
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        return True
                
                cap.release()
                return False
            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)
                os.close(devnull)
        except:
            return False
    
    @classmethod
    def discover_cameras(cls) -> List[Dict]:
        cls._suppress_errors()
        cameras = []
        
        for i in range(10):
            if is_shutdown():
                break
            
            device_path = f'/dev/video{i}'
            if not Path(device_path).exists():
                continue
            
            name = f"Camera {i}"
            can_capture = cls._test_capture(i)
            
            if can_capture:
                cameras.append({
                    'device': i,
                    'path': device_path,
                    'name': name,
                    'tested': can_capture
                })
        
        return cameras

# ==========================================================================
# Video Source Class
# ==========================================================================
class AdvantechVideoSource:
    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_type = "unknown"
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
        self.is_video_file = False
        self._frame_count = 0
        self._start_time = 0
    
    def open_camera(self, device: int, width: int = 1280, height: int = 720,
                   pixel_format: str = "MJPG", fps: int = 30) -> bool:
        if is_shutdown():
            return False
        
        print(f"Attempting to open camera device: {device}")
        print(f"Parameters: {width}x{height}, format: {pixel_format}, FPS: {fps}")
        
        methods = [
            self._try_open_direct,
            self._try_open_v4l2,
            self._try_open_gstreamer,
            self._try_open_simple
        ]
        
        for method in methods:
            if is_shutdown():
                return False
            
            try:
                print(f"Trying method {method.__name__}...")
                success = method(device, width, height, pixel_format, fps)
                if success and self._cap and self._cap.isOpened():
                    self._source_type = "camera"
                    
                    actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
                    
                    self.width = actual_width if actual_width > 0 else width
                    self.height = actual_height if actual_height > 0 else height
                    self.fps = actual_fps if actual_fps > 0 else fps
                    self._start_time = time.time()
                    
                    print(f"Camera opened successfully!")
                    print(f"Actual resolution: {self.width}x{self.height}, FPS: {self.fps}")
                    return True
            except Exception as e:
                print(f"Method {method.__name__} failed: {e}")
                continue
        
        print("All camera opening methods failed")
        return False
    
    def _try_open_direct(self, device, width, height, pixel_format, fps):
        try:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                return False

            # Must be set before read()
            fourcc = cv2.VideoWriter_fourcc(*pixel_format[:4])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            # Force trigger format negotiation
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False

            self._cap = cap
            return True
        except Exception as e:
            print("direct open failed:", e)
            return False
    
    def _try_open_v4l2(self, device: int, width: int, height: int,
                      pixel_format: str, fps: int) -> bool:
        try:
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if self._cap.isOpened():
                if pixel_format.upper() in ['MJPG', 'MJPEG']:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return True
                else:
                    self._cap.release()
                    self._cap = None
        except:
            pass
        return False
    
    def _try_open_gstreamer(self, device: int, width: int, height: int,
                           pixel_format: str, fps: int) -> bool:
        try:
            if device == -1:
                gst = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={width},height={height},"
                       f"framerate={fps}/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! "
                       f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false")
                self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            elif pixel_format.upper() in ['MJPG', 'MJPEG']:
                gst = (f"v4l2src device=/dev/video{device} ! "
                       f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                       f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1")
                self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            else:
                gst = (f"v4l2src device=/dev/video{device} ! "
                       f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
                       f"videoconvert ! appsink drop=1")
                self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return True
                else:
                    self._cap.release()
                    self._cap = None
        except:
            pass
        return False
    
    def _try_open_simple(self, device: int, width: int, height: int,
                        pixel_format: str, fps: int) -> bool:
        try:
            if device >= 0:
                device_path = f"/dev/video{device}"
                print(f"Trying to open device path: {device_path}")
                self._cap = cv2.VideoCapture(device_path)
                if self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        print(f"Successfully opened via device path: {device_path}")
                        return True
                    else:
                        self._cap.release()
                        self._cap = None
        except:
            pass
        
        try:
            self._cap = cv2.VideoCapture(device)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    print("Successfully opened camera with minimal parameters")
                    return True
                else:
                    self._cap.release()
                    self._cap = None
        except:
            pass
        
        return False
    
    def open_file(self, path: str) -> bool:
        if is_shutdown():
            return False
        
        if not Path(path).exists():
            print(f"File does not exist: {path}")
            return False
        
        try:
            self._cap = cv2.VideoCapture(path)
            if self._cap.isOpened():
                self._source_type = "file"
                self.is_video_file = True
                self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self._cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 30
                
                self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._start_time = time.time()
                print(f"Video info: {self.width}x{self.height}, FPS: {self.fps:.2f}")
                return True
            else:
                print(f"Cannot open video file: {path}")
        except Exception as e:
            print(f"Failed to open video file: {e}")
        
        return False
    
    def read(self) -> Optional[np.ndarray]:
        if is_shutdown() or self._cap is None:
            return None
        
        try:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._frame_count += 1
                return frame
        except:
            pass
        
        return None
    
    def release(self):
        if self._cap:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None

# ==========================================================================
# AMD NPU Detection Functions
# ==========================================================================
def detect_amd_npu() -> Dict[str, Any]:
    """
    Detect AMD NPU availability
    Returns: {
        'available': bool,
        'type': str,  # 'PHX/HPT', 'STX', 'KRK', 'HPT-XDNA2', or 'UNKNOWN'
        'xclbin_path': str,
        'message': str
    }
    """
    npu_info = {
        'available': False,
        'type': 'UNKNOWN',
        'xclbin_path': '',
        'message': 'No AMD NPU detected'
    }
    
    try:
        # Method 1: Check Vitis AI environment variables
        vitisai_env = os.environ.get('VITIS_AI_HOME', '')
        if vitisai_env and os.path.exists(vitisai_env):
            npu_info['available'] = True
            npu_info['message'] = f'Vitis AI environment detected: {vitisai_env}'
            
            # Try to determine NPU type
            # First, check CPU model for specific Ryzen processors
            cpu_model = _get_cpu_model_name()
            
            # Check device files
            if os.path.exists('/dev/xclmgmt*') or os.path.exists('/dev/dri/renderD*'):
                # Determine NPU type based on CPU model and device files
                if cpu_model:
                    # Check for AMD Ryzen R7 8845HS (Hawk Point with XDNA2 NPU)
                    if '8845hs' in cpu_model.lower():
                        npu_info['type'] = 'HPT-XDNA2'
                        npu_info['message'] = f'AMD Ryzen R7 8845HS detected (HPT-XDNA2 NPU)'
                    # Check for other Hawk Point processors
                    elif any(x in cpu_model.lower() for x in ['hawk point', 'hpt', 'ryzen ai']):
                        npu_info['type'] = 'HPT'
                        npu_info['message'] = f'AMD {cpu_model} detected (HPT NPU)'
                    # Check for Phoenix processors
                    elif any(x in cpu_model.lower() for x in ['phoenix', 'phx']):
                        npu_info['type'] = 'PHX'
                        npu_info['message'] = f'AMD {cpu_model} detected (PHX NPU)'
                    else:
                        # Default to PHX/HPT for generic XDNA-based NPUs
                        npu_info['type'] = 'PHX/HPT'
                        npu_info['message'] = f'AMD NPU detected, CPU: {cpu_model}'
                else:
                    # Couldn't determine CPU model, use default type
                    npu_info['type'] = 'PHX/HPT'
                
                # Find xclbin file
                xclbin_search_paths = [
                    '/opt/xilinx/xrt/share/xclbin',
                    '/usr/lib/xrt/share/xclbin',
                    str(Path.home() / '.xclbin'),
                    os.path.join(vitisai_env, 'xclbin')
                ]
                for path in xclbin_search_paths:
                    if os.path.exists(path):
                        for file in os.listdir(path):
                            if file.endswith('.xclbin'):
                                npu_info['xclbin_path'] = os.path.join(path, file)
                                break
                        if npu_info['xclbin_path']:
                            break
            elif os.path.exists('/sys/class/vai'):
                npu_info['type'] = 'STX/KRK'
                npu_info['message'] = 'AMD STX/KRK NPU detected'
        
        # Method 2: Check if Vitis AI Runtime is installed
        try:
            result = subprocess.run(['dpkg', '-l', 'xrt'], 
                                  capture_output=True, text=True)
            if 'xrt' in result.stdout:
                npu_info['available'] = True
                if not npu_info['type'] or npu_info['type'] == 'UNKNOWN':
                    # Try to get CPU model for better type detection
                    cpu_model = _get_cpu_model_name()
                    if cpu_model and '8845hs' in cpu_model.lower():
                        npu_info['type'] = 'HPT-XDNA2'
                        npu_info['message'] = f'AMD Ryzen R7 8845HS detected via XRT (HPT-XDNA2 NPU)'
                    else:
                        npu_info['type'] = 'PHX/HPT'
                        npu_info['message'] = 'XRT runtime detected (AMD NPU)'
        except:
            pass
        
        # Method 3: Check if ONNX Runtime supports Vitis AI EP
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'VitisAIExecutionProvider' in providers:
                npu_info['available'] = True
                if not npu_info['type'] or npu_info['type'] == 'UNKNOWN':
                    # Try to get CPU model for better type detection
                    cpu_model = _get_cpu_model_name()
                    if cpu_model and '8845hs' in cpu_model.lower():
                        npu_info['type'] = 'HPT-XDNA2'
                        npu_info['message'] = f'AMD Ryzen R7 8845HS detected via ONNX Runtime (HPT-XDNA2 NPU)'
                    else:
                        npu_info['type'] = 'PHX/HPT'
                        npu_info['message'] = 'ONNX Runtime Vitis AI Execution Provider detected'
                else:
                    npu_info['message'] = 'ONNX Runtime Vitis AI Execution Provider detected'
        except:
            pass
            
    except Exception as e:
        npu_info['message'] = f'NPU detection failed: {str(e)}'
    
    return npu_info

def _get_cpu_model_name() -> str:
    """Get CPU model name from system"""
    try:
        # Method 1: Read from /proc/cpuinfo
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('model name'):
                        return line.split(':')[1].strip()
        except:
            pass
        
        # Method 2: Use lscpu command
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Model name' in line:
                    return line.split(':')[1].strip()
        except:
            pass
        
        # Method 3: Use dmidecode command (may require root)
        try:
            result = subprocess.run(['dmidecode', '-t', 'processor'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Version:' in line and 'AMD' in line:
                    return line.split('Version:')[1].strip()
        except:
            pass
        
        return ""
    except:
        return ""

# ==========================================================================
# Enhanced Multi-threaded Pipeline - Supports AMD NPU Status Display and Performance Monitoring
# ==========================================================================
class AdvantechPipeline:
    def __init__(self, config: AdvantechConfig, logger: AdvantechLogger, 
                 source: AdvantechVideoSource, engine: AdvantechEngine,
                 npu_info: Dict[str, Any] = None):
        self.config = config
        self.logger = logger
        self.source = source
        self.engine = engine
        self.npu_info = npu_info or {}
        
        self._running = False
        self._stop_event = threading.Event()
        
        self._capture_thread = None
        self._output_thread = None
        self._frame_buffer = None
        self._frame_lock = threading.Lock()
        self._result_buffer = None
        self._result_lock = threading.Lock()
        
        self._total_frames = 0
        self._processed_frames = 0
        self._start_time = 0
        
        self._encoder = None
        self._renderer = AdvantechOverlayRenderer(config)
        
        self._window_created = False
        self._window_closed = False
        
        self._max_buffer_size = 5
        

        self._performance_history = []
        self._last_performance_update = 0
        self._performance_update_interval = 1.0  
    
    def start(self):
        self._running = True
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._window_closed = False
        
        if self.config.save_video and self.config.output_path:
            self._init_video_writer()
        
        # Display device information
        device_type = getattr(self.engine, 'device_type', 'CPU')
        if device_type == 'AMD NPU':
            print(f"\n🚀 Using AMD NPU acceleration: {self.npu_info.get('type', 'UNKNOWN')}")
            print(f"   NPU Info: {self.npu_info.get('message', '')}")
        else:
            print(f"\n💻 Using CPU inference")
            if self.npu_info.get('available', False):
                print(f"   ⚠️  NPU available but not used, please check model compatibility or configuration")
        
        # Display task information
        task_name = self.config.task_type.value.upper()
        print(f"Starting {task_name} task...")
        print(f"Resolution: {self.source.width}x{self.source.height}")
        print(f"FPS: {self.source.fps:.2f}")
        

        print(f"\n📊 Performance monitoring enabled")
        print(f"   Press 'p' to show detailed performance metrics")
        print(f"   Press 's' to save performance report")
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._output_thread = threading.Thread(target=self._output_loop, daemon=True, name="OutputThread")
        
        self._capture_thread.start()
        self._output_thread.start()
        
        try:
            self._inference_loop()
        except Exception as e:
            print(f"Inference loop exception: {e}")
            import traceback
            traceback.print_exc()
            self._stop_event.set()
    
    def _init_video_writer(self):
        try:
            output_path = self.config.output_path
            if os.path.isdir(output_path):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_path, f"output_{timestamp}.mp4")
            
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.source.fps if self.source.fps > 0 else 30
            self._encoder = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps, 
                (self.source.width, self.source.height)
            )
            
            if self._encoder.isOpened():
                print(f"Video saving to: {output_path}")
            else:
                print(f"Warning: Cannot create video writer")
                self._encoder = None
        except Exception as e:
            print(f"Failed to initialize video writer: {e}")
            self._encoder = None
    
    def _capture_loop(self):
        frame_interval = 1.0 / (self.source.fps or 30) if self.source.is_video_file else 0
        last_time = time.perf_counter()
        
        while self._running and not self._stop_event.is_set() and not is_shutdown():
            frame = self.source.read()
            if frame is None:
                if self.source.is_video_file:
                    print("Video file reading completed")
                    self._stop_event.set()
                    break
                continue
            
            if self.source.is_video_file and frame_interval > 0:
                elapsed = time.perf_counter() - last_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_time = time.perf_counter()
            
            with self._frame_lock:
                self._frame_buffer = (frame, time.perf_counter())
    
    def _inference_loop(self):
        while self._running and not self._stop_event.is_set() and not is_shutdown():
            frame_data = None
            with self._frame_lock:
                if self._frame_buffer is not None:
                    frame_data = self._frame_buffer
                    self._frame_buffer = None
            
            if frame_data is None:
                time.sleep(0.001)
                continue
            
            frame, timestamp = frame_data
            
            try:
                result = self.engine.infer(frame)
                
                with self._result_lock:
                    self._result_buffer = {
                        'frame': frame,
                        'result': result,
                        'timestamp': timestamp
                    }
                
                self._processed_frames += 1
                
            except Exception as e:
                print(f"Inference failed: {e}")
                continue
    
    def _check_window_closed(self):
        if not self._window_created:
            return False
        
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) <= 0:
                self._window_closed = True
                return True
        except:
            self._window_closed = True
            return True
        
        return False
    
    def _output_loop(self):
        self._window_created = False
        show_performance_details = False
        
        while self._running and not self._stop_event.is_set() and not is_shutdown():
            if self._window_created and self._check_window_closed():
                print("Window closed, stopping processing...")
                self._stop_event.set()
                request_shutdown()
                break
            
            result_data = None
            with self._result_lock:
                if self._result_buffer is not None:
                    result_data = self._result_buffer
                    self._result_buffer = None
            
            if result_data is None:
                if self.config.show_display and self._window_created:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("User requested exit (pressed q)")
                        self._stop_event.set()
                        request_shutdown()
                        break
                    elif key == ord('p'):
                        show_performance_details = not show_performance_details
                        print(f"Performance details: {'ON' if show_performance_details else 'OFF'}")
                    elif key == ord('s'):
                        self._save_performance_report()
                time.sleep(0.001)
                continue
            
            self._total_frames += 1
            
            frame = result_data['frame']
            result = result_data['result']
            
            elapsed = time.perf_counter() - self._start_time
            fps = self._total_frames / elapsed if elapsed > 0 else 0
            
            metrics = AdvantechMetrics(
                fps=fps,
                total_frames=self._total_frames
            )
            

            performance_metrics = None
            if hasattr(self.engine, 'get_performance_metrics'):
                performance_metrics = self.engine.get_performance_metrics()
                

                current_time = time.perf_counter()
                if current_time - self._last_performance_update >= self._performance_update_interval:
                    self._performance_history.append({
                        'timestamp': current_time,
                        'fps': fps,
                        'latency_ms': performance_metrics.latency_ms,
                        'memory_mb': performance_metrics.memory_usage_mb
                    })

                    if len(self._performance_history) > 100:
                        self._performance_history.pop(0)
                    self._last_performance_update = current_time
            
            try:
                if isinstance(result, AdvantechClassification):
                    # Classification task
                    frame_display = self._renderer.render(frame, [], metrics, result, performance_metrics)
                elif isinstance(result, list):
                    # Detection or segmentation task
                    frame_display = self._renderer.render(frame, result, metrics, None, performance_metrics)
                else:
                    # No results
                    frame_display = frame
            except Exception as e:
                print(f"Rendering failed: {e}")
                frame_display = frame
            

            if show_performance_details and performance_metrics:
                self._add_performance_details(frame_display, performance_metrics)
            

            device_type = getattr(self.engine, 'device_type', 'CPU')
            if device_type == 'AMD NPU':
                device_text = f"AMD NPU ({self.npu_info.get('type', 'UNKNOWN')})"
                color = (0, 255, 0)  # Green
            else:
                device_text = "CPU"
                color = (255, 0, 0)  # Red
            
            # cv2.putText(frame_display, f"Device: {device_text}", (10, frame_display.shape[0] - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            

            cv2.putText(frame_display, "Press 'p' for performance details", 
                       (frame_display.shape[1] - 300, frame_display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if self._encoder:
                try:
                    self._encoder.write(frame_display)
                except Exception as e:
                    print(f"Failed to save video frame: {e}")
            
            if self.config.show_display:
                try:
                    if not self._window_created:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
                        self._window_created = True
                    
                    cv2.imshow(WINDOW_NAME, frame_display)
                    
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) <= 0:
                        print("Window close button clicked")
                        self._stop_event.set()
                        request_shutdown()
                        break
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("User requested exit (pressed q)")
                        self._stop_event.set()
                        request_shutdown()
                        break
                    elif key == ord('p'):
                        show_performance_details = not show_performance_details
                        print(f"Performance details: {'ON' if show_performance_details else 'OFF'}")
                    elif key == ord('s'):
                        self._save_performance_report()
                except Exception as e:
                    print(f"Display failed: {e}")
            
            if self._total_frames % 30 == 0:
                elapsed_time = time.perf_counter() - self._start_time
                current_fps = self._total_frames / elapsed_time if elapsed_time > 0 else 0
                task_name = self.config.task_type.value.upper()
                

                perf_info = ""
                if performance_metrics:
                    perf_info = f", Latency: {performance_metrics.latency_ms:.1f}ms, Memory: {performance_metrics.memory_usage_mb:.0f}MB"
                
                print(f"[{task_name}] Processed {self._total_frames} frames, FPS: {current_fps:.2f}, Device: {device_type}{perf_info}")
    
    def _add_performance_details(self, frame: np.ndarray, metrics: AdvantechPerformanceMetrics):

        height, width = frame.shape[:2]
        

        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 350, 50), (width - 10, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        

        cv2.putText(frame, "Performance Details:", (width - 340, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        

        device_color = (0, 255, 0) if "NPU" in metrics.device_type.upper() else (255, 0, 0)
        cv2.putText(frame, f"Device: {metrics.device_type}", (width - 340, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, device_color, 1)
        

        y_pos = 140
        lines = [
            f"FPS: {metrics.fps:.1f}",
            f"Latency: {metrics.latency_ms:.1f}ms",
            f"Inference: {metrics.avg_inference_time_ms:.1f}ms",
            f"Preprocess: {metrics.avg_preprocess_time_ms:.1f}ms",
            f"Postprocess: {metrics.avg_postprocess_time_ms:.1f}ms",
            f"Memory: {metrics.memory_usage_mb:.0f}MB",
            f"CPU: {metrics.cpu_usage_percent:.1f}%"
        ]
        
        for line in lines:
            cv2.putText(frame, line, (width - 340, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20
    
    def _save_performance_report(self):

        if hasattr(self.engine, 'get_performance_metrics'):
            metrics = self.engine.get_performance_metrics()
            

            report = {
                'timestamp': datetime.now().isoformat(),
                'device_type': metrics.device_type,
                'model_name': metrics.model_name,
                'total_frames': self._total_frames,
                'total_time_seconds': time.perf_counter() - self._start_time,
                'avg_fps': metrics.fps,
                'avg_latency_ms': metrics.latency_ms,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'performance_history': self._performance_history[-50:]  # 最近50条记录
            }
            

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("performance_reports")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f"performance_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📊 Performance report saved: {report_file}")
    
    def stop(self):
        print("Stopping processing pipeline...")
        self._running = False
        self._stop_event.set()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._output_thread:
            self._output_thread.join(timeout=1.0)
        
        self.source.release()
        
        if self._encoder:
            try:
                self._encoder.release()
                print("Video writer released")
            except Exception as e:
                print(f"Failed to release video writer: {e}")
            self._encoder = None
        
        try:
            if self._window_created:
                cv2.destroyWindow(WINDOW_NAME)
            cv2.destroyAllWindows()
        except:
            pass
        
        self._window_created = False
        self._window_closed = False
        
        with self._frame_lock:
            self._frame_buffer = None
        with self._result_lock:
            self._result_buffer = None
        

        self._save_performance_report()
    
    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set() and not is_shutdown()
    
    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self._start_time
        avg_fps = self._total_frames / elapsed if elapsed > 0 else 0
        
        device_type = getattr(self.engine, 'device_type', 'CPU')
        

        performance_metrics = None
        if hasattr(self.engine, 'get_performance_metrics'):
            performance_metrics = self.engine.get_performance_metrics()
        
        stats = {
            'total_frames': self._total_frames,
            'processed_frames': self._processed_frames,
            'avg_fps': avg_fps,
            'elapsed_time': elapsed,
            'device_type': device_type,
            'task_type': self.config.task_type.value
        }
        
        if performance_metrics:
            stats.update({
                'performance_metrics': performance_metrics.to_dict(),
                'latency_ms': performance_metrics.latency_ms,
                'memory_usage_mb': performance_metrics.memory_usage_mb,
                'cpu_usage_percent': performance_metrics.cpu_usage_percent
            })
        
        return stats

# ==========================================================================
# Enhanced Pipeline for Video File Comparison
# ==========================================================================
class AdvantechVideoComparisonPipeline:

    
    def __init__(self, config: AdvantechConfig, logger: AdvantechLogger,
                 video_path: str, output_path: str = None):
        self.config = config
        self.logger = logger
        self.video_path = video_path
        self.output_path = output_path
        
        self._npu_engine = None
        self._cpu_engine = None
        self._video_source = None
        
        self._npu_metrics = AdvantechPerformanceMetrics()
        self._cpu_metrics = AdvantechPerformanceMetrics()
        
        self._npu_results = []
        self._cpu_results = []
    
    def run_comparison(self) -> AdvantechPerformanceComparison:

        print_banner()
        

        if not Path(self.video_path).exists():
            print(f"Error: Video file does not exist: {self.video_path}")
            return None
        

        self._video_source = AdvantechVideoSource()
        if not self._video_source.open_file(self.video_path):
            print(f"Error: Cannot open video file: {self.video_path}")
            return None
        
        print(f"\n📹 Video file info:")
        print(f"   Resolution: {self._video_source.width}x{self._video_source.height}")
        print(f"   FPS: {self._video_source.fps:.2f}")
        print(f"   Total frames: {self._video_source.total_frames}")
        

        if self.output_path:
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        

        print(f"\n{'='*80}")
        print("RUNNING NPU PERFORMANCE TEST")
        print('='*80)
        self.config.use_npu = True
        npu_result = self._run_video_test(device_type="NPU")
        
        if npu_result is None:
            print("NPU test failed, cannot continue comparison")
            return None
        
        self._npu_metrics = npu_result['metrics']
        self._npu_results = npu_result['results']
        

        self._video_source.release()
        self._video_source.open_file(self.video_path)
        

        print(f"\n{'='*80}")
        print("RUNNING CPU PERFORMANCE TEST")
        print('='*80)
        self.config.use_npu = False
        cpu_result = self._run_video_test(device_type="CPU")
        
        if cpu_result is None:
            print("CPU test failed, cannot complete comparison")
            return None
        
        self._cpu_metrics = cpu_result['metrics']
        self._cpu_results = cpu_result['results']
        

        comparison = AdvantechPerformanceComparison(
            npu_metrics=self._npu_metrics,
            cpu_metrics=self._cpu_metrics,
            test_duration_seconds=max(self._npu_metrics.total_time_ms, 
                                    self._cpu_metrics.total_time_ms) / 1000
        )
        

        print(comparison.get_comparison_summary())
        

        self._save_detailed_comparison_report(comparison)
        
        return comparison
    
    def _run_video_test(self, device_type: str) -> Dict[str, Any]:

        print(f"\nRunning {device_type} test...")
        

        loader = AdvantechModelLoader(self.config, self.logger)
        force_cpu = (device_type == "CPU")
        
        try:
            engine = loader.load(self.config.model_path, force_cpu=force_cpu)
        except Exception as e:
            print(f"Failed to load {device_type} engine: {e}")
            return None
        

        print(f"Warming up {device_type} engine (5 frames)...")
        warmup_frames = []
        for i in range(5):
            if is_shutdown():
                break
            frame = self._video_source.read()
            if frame is None:
                break
            warmup_frames.append(frame)
        
        for frame in warmup_frames:
            engine.infer(frame)
        

        self._video_source.release()
        self._video_source.open_file(self.video_path)
        

        print(f"Running {device_type} test on full video...")
        start_time = time.perf_counter()
        
        results = []
        frame_count = 0
        last_print_time = start_time
        
        while True:
            if is_shutdown():
                break
                
            frame = self._video_source.read()
            if frame is None:
                break
            

            inference_start = time.perf_counter()
            result = engine.infer(frame)
            inference_time = (time.perf_counter() - inference_start) * 1000
            

            results.append({
                'frame_id': frame_count,
                'inference_time_ms': inference_time,
                'result': result
            })
            
            frame_count += 1
            

            current_time = time.perf_counter()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  {device_type}: Processed {frame_count}/{self._video_source.total_frames} frames, FPS: {fps:.1f}")
                last_print_time = current_time
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        

        metrics = engine.get_performance_metrics()
        metrics.device_type = device_type
        metrics.total_frames = frame_count
        metrics.total_time_ms = total_time * 1000
        
        if total_time > 0:
            metrics.fps = frame_count / total_time
            metrics.throughput_fps = metrics.fps
            metrics.latency_ms = metrics.avg_total_time_ms
        

        engine.cleanup()
        
        print(f"  {device_type} test completed:")
        print(f"    Total frames: {frame_count}")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Average FPS: {metrics.fps:.2f}")
        print(f"    Average latency: {metrics.avg_total_time_ms:.2f}ms")
        
        return {
            'metrics': metrics,
            'results': results
        }
    
    def _save_detailed_comparison_report(self, comparison: AdvantechPerformanceComparison):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("performance_comparison_reports")
        report_dir.mkdir(exist_ok=True)
        

        txt_report = report_dir / f"video_comparison_{timestamp}.txt"
        json_report = report_dir / f"video_comparison_{timestamp}.json"
        
        comparison.save_report(str(txt_report))
        comparison.to_json(str(json_report))
        

        detailed_report = {
            'video_info': {
                'path': self.video_path,
                'resolution': f"{self._video_source.width}x{self._video_source.height}",
                'fps': self._video_source.fps,
                'total_frames': self._video_source.total_frames
            },
            'comparison': {
                'timestamp': datetime.now().isoformat(),
                'speedup_ratio': comparison.speedup_ratio,
                'efficiency_gain': comparison.efficiency_gain,
                'energy_savings_percent': comparison.energy_savings_percent,
                'recommendation': comparison.recommendation
            },
            'npu_performance': self._npu_metrics.to_dict(),
            'cpu_performance': self._cpu_metrics.to_dict(),
            'frame_by_frame_comparison': []
        }
        

        sample_rate = max(1, len(self._npu_results) // 100)  
        for i in range(0, min(len(self._npu_results), len(self._cpu_results)), sample_rate):
            npu_frame = self._npu_results[i]
            cpu_frame = self._cpu_results[i]
            
            detailed_report['frame_by_frame_comparison'].append({
                'frame_id': i,
                'npu_inference_ms': npu_frame['inference_time_ms'],
                'cpu_inference_ms': cpu_frame['inference_time_ms'],
                'speedup': cpu_frame['inference_time_ms'] / npu_frame['inference_time_ms'] if npu_frame['inference_time_ms'] > 0 else 0
            })
        
        detailed_json = report_dir / f"video_comparison_detailed_{timestamp}.json"
        with open(detailed_json, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\n📊 Performance comparison reports saved:")
        print(f"  Summary report: {txt_report}")
        print(f"  JSON report: {json_report}")
        print(f"  Detailed report: {detailed_json}")
        

        self._create_visualization_report(detailed_report, report_dir, timestamp)
    
    def _create_visualization_report(self, detailed_report: Dict, report_dir: Path, timestamp: str):

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg') 
            

            frames = []
            npu_times = []
            cpu_times = []
            
            for frame_data in detailed_report['frame_by_frame_comparison']:
                frames.append(frame_data['frame_id'])
                npu_times.append(frame_data['npu_inference_ms'])
                cpu_times.append(frame_data['cpu_inference_ms'])
            
            if frames:

                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                

                ax1 = axes[0, 0]
                ax1.plot(frames, npu_times, 'b-', label='NPU', alpha=0.7)
                ax1.plot(frames, cpu_times, 'r-', label='CPU', alpha=0.7)
                ax1.set_xlabel('Frame Number')
                ax1.set_ylabel('Inference Time (ms)')
                ax1.set_title('Frame-by-Frame Inference Time Comparison')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                

                ax2 = axes[0, 1]
                speedup_ratios = [c/n if n > 0 else 0 for c, n in zip(cpu_times, npu_times)]
                ax2.plot(frames, speedup_ratios, 'g-', alpha=0.7)
                ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
                ax2.set_xlabel('Frame Number')
                ax2.set_ylabel('Speedup Ratio (CPU/NPU)')
                ax2.set_title('Speedup Ratio per Frame')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                

                ax3 = axes[1, 0]
                sorted_npu = sorted(npu_times)
                sorted_cpu = sorted(cpu_times)
                p_npu = 100. * np.arange(len(sorted_npu)) / (len(sorted_npu) - 1)
                p_cpu = 100. * np.arange(len(sorted_cpu)) / (len(sorted_cpu) - 1)
                ax3.plot(sorted_npu, p_npu, 'b-', label='NPU')
                ax3.plot(sorted_cpu, p_cpu, 'r-', label='CPU')
                ax3.set_xlabel('Inference Time (ms)')
                ax3.set_ylabel('Cumulative Percentage (%)')
                ax3.set_title('Cumulative Distribution of Inference Times')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                

                ax4 = axes[1, 1]
                data = [npu_times, cpu_times]
                bp = ax4.boxplot(data, labels=['NPU', 'CPU'], patch_artist=True)
                

                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax4.set_ylabel('Inference Time (ms)')
                ax4.set_title('Distribution Comparison (Box Plot)')
                ax4.grid(True, alpha=0.3)
                
                plt.suptitle('NPU vs CPU Performance Comparison', fontsize=16, fontweight='bold')
                plt.tight_layout()
                

                plot_file = report_dir / f"video_comparison_plot_{timestamp}.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Visualization plot: {plot_file}")
                
        except ImportError:
            print("  Note: matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"  Note: Failed to create visualization: {e}")

# ==========================================================================
# Print Program Banner
# ==========================================================================
def print_banner():
    banner=f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗  ║
║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║  ║
║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║  ║
║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║  ║
║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║  ║
║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝  ║
║                        YOLO Inference Pipeline v{__version__}                            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Author: {__author__:<18}  Build: {__build_date__:<14}                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║       {__copyright__}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""
    print(banner)

# ==========================================================================
# Command Line Interface Class (Enhanced with Video Comparison)
# ==========================================================================
class AdvantechCLI:
    def __init__(self):
        self.config = AdvantechConfig()
        self.logger = AdvantechLogger(self.config)
        self._pipeline: Optional[AdvantechPipeline] = None
        self.npu_info = detect_amd_npu()
        

        self.performance_comparer = None
        if PERFORMANCE_FEATURES_AVAILABLE:
            self.performance_comparer = AdvantechPerformanceComparer(self.config, self.logger)
    
    def list_cameras(self):
        print_banner()
        print("\nDetecting cameras...")
        
        cameras = AdvantechCameraDiscovery.discover_cameras()
        
        if not cameras:
            print("\nNo cameras detected")
            print("\nTroubleshooting:")
            print("  1. Check camera connection: ls -la /dev/video*")
            print("  2. Check permissions: sudo chmod 666 /dev/video*")
            print("  3. Test camera: v4l2-ctl --list-devices")
            return
        
        print(f"\nDetected {len(cameras)} cameras:\n")
        
        for cam in cameras:
            status = "[Working]" if cam.get('tested', False) else "[Unknown]"
            print(f"  {status} [{cam['device']}] {cam['name']}")
            print(f"      Path: {cam['path']}")
            print()
    
    def _get_choice(self, prompt: str, choices: List[str], default: str) -> str:
        while True:
            try:
                choice = input(f"{prompt} [{'/'.join(choices)}] (default: {default}): ").strip()
                if not choice:
                    return default
                if choice in choices:
                    return choice
                print(f"Invalid choice, please choose from {choices}")
            except (EOFError, KeyboardInterrupt):
                return default
    
    def _get_input(self, prompt: str, default: str) -> str:
        try:
            value = input(f"{prompt} (default: {default}): ").strip()
            return value if value else default
        except (EOFError, KeyboardInterrupt):
            return default
    
    def run_interactive(self):
        print_banner()
        

        if self.npu_info['available']:
            print(f"\n✅ AMD NPU detected: {self.npu_info['type']}")
            print(f"   Info: {self.npu_info['message']}")
            

            compare_choice = self._get_choice("Run video performance comparison (NPU vs CPU)? (y/n)", ["y", "n", "Y", "N"], "n")
            if compare_choice.lower() == "y":
                self._run_video_comparison_interactive()
                return
            

            use_npu_choice = self._get_choice("Use NPU acceleration? (y/n)", ["y", "n", "Y", "N"], "y")
            self.config.use_npu = use_npu_choice.lower() == "y"
            if not self.config.use_npu:
                print(f"   ⚠️  NPU will NOT be used (CPU-only mode)")
        else:
            print(f"\n⚠️  No AMD NPU detected, will use CPU inference")
            print(f"   Info: {self.npu_info['message']}")
        
        print("\n[1] Detection  [2] Classification  [3] Segmentation")
        task_choice = self._get_choice("Select task", ["1", "2", "3"], "1")
        self.config.task_type = {
            "1": AdvantechTaskType.DETECTION,
            "2": AdvantechTaskType.CLASSIFICATION,
            "3": AdvantechTaskType.SEGMENTATION
        }[task_choice]
        
        print("\n[1] ONNX  [2] PyTorch")
        format_choice = self._get_choice("Select model format", ["1", "2"], "1")
        self.config.model_format = {
            "1": AdvantechModelFormat.ONNX,
            "2": AdvantechModelFormat.PYTORCH
        }[format_choice]
        
        if format_choice == "1" and not ONNX_AVAILABLE:
            print("[WARNING] ONNX Runtime is not available. Please install it first.")
            print("[INFO] Install: pip install onnxruntime")
            return
        
        if format_choice == "2" and not TORCH_AVAILABLE:
            print("[WARNING] PyTorch is not available. Please install it first.")
            print("[INFO] Install: pip install torch torchvision")
            return
        
        # Suggest model based on task
        model_suggestions = {
            AdvantechTaskType.DETECTION: "yolo11n.onnx",
            AdvantechTaskType.CLASSIFICATION: "yolo11n-cls.onnx",
            AdvantechTaskType.SEGMENTATION: "yolo11n-seg.onnx"
        }
        suggestion = model_suggestions.get(self.config.task_type, "yolo11n.onnx")
        
        self.config.model_path = self._get_input(f"Model path ({suggestion})", suggestion)
        if not self.config.model_path:
            print("Error: Must specify model path")
            return
        
        print("\n[1] Camera  [2] RTSP  [3] Video File")
        source_choice = self._get_choice("Select input source", ["1", "2", "3"], "1")
        self.config.input_source = {
            "1": AdvantechInputSource.CAMERA,
            "2": AdvantechInputSource.RTSP,
            "3": AdvantechInputSource.FILE
        }[source_choice]
        
        if self.config.input_source == AdvantechInputSource.CAMERA:
            cameras = AdvantechCameraDiscovery.discover_cameras()
            if cameras:
                print("\nAvailable cameras:")
                for cam in cameras:
                    status = "[OK]" if cam.get('tested', False) else "[?]"
                    print(f"  {status} [{cam['device']}] {cam['name']}")
            else:
                print("\nNo cameras automatically detected, you can still enter device manually")
                print("  Check: ls /dev/video*")
            device = self._get_input("Camera device (0, 1, etc.)", "0")
            self.config.camera_device = device if device.startswith('/dev/') else f"/dev/video{device}"
            self.config.camera_width = 1280
            self.config.camera_height = 720
            self.config.camera_format = "MJPG"
            self.config.camera_fps = 30
        elif self.config.input_source == AdvantechInputSource.RTSP:
            self.config.input_path = self._get_input("RTSP URL", "rtsp://192.168.1.100:554/stream")
        else:
            self.config.input_path = self._get_input("Video file path", "input.mp4")
        
        save_choice = self._get_choice("Save output video? (y/n)", ["y", "n", "Y", "N"], "n")
        self.config.save_video = save_choice.lower() == "y"
        if self.config.save_video:
            self.config.output_path = self._get_input("Output path", "./output.mp4")
        
        display_choice = self._get_choice("Show results? (y/n)", ["y", "n", "Y", "N"], "y")
        self.config.show_display = display_choice.lower() == "y"
        

        monitor_choice = self._get_choice("Enable performance monitoring? (y/n)", ["y", "n", "Y", "N"], "y")
        self.config.compare_performance = monitor_choice.lower() == "y"
        
        self._run_pipeline()
    
    def _run_video_comparison_interactive(self):

        print("\n" + "="*80)
        print("VIDEO PERFORMANCE COMPARISON SETUP")
        print("="*80)
        
        print("\n[1] Detection  [2] Classification  [3] Segmentation")
        task_choice = self._get_choice("Select task for comparison", ["1", "2", "3"], "1")
        self.config.task_type = {
            "1": AdvantechTaskType.DETECTION,
            "2": AdvantechTaskType.CLASSIFICATION,
            "3": AdvantechTaskType.SEGMENTATION
        }[task_choice]
        
        # Suggest model based on task
        model_suggestions = {
            AdvantechTaskType.DETECTION: "yolo11n.onnx",
            AdvantechTaskType.CLASSIFICATION: "yolo11n-cls.onnx",
            AdvantechTaskType.SEGMENTATION: "yolo11n-seg.onnx"
        }
        suggestion = model_suggestions.get(self.config.task_type, "yolo11n.onnx")
        
        self.config.model_path = self._get_input(f"Model path for comparison ({suggestion})", suggestion)
        if not self.config.model_path:
            print("Error: Must specify model path for comparison")
            return
        

        self.config.input_path = self._get_input("Video file path for comparison", "test.mp4")
        

        save_choice = self._get_choice("Save output videos for comparison? (y/n)", ["y", "n", "Y", "N"], "n")
        save_videos = save_choice.lower() == "y"
        
        output_path = None
        if save_videos:
            output_path = self._get_input("Output video path (will append _npu and _cpu)", "./comparison_output.mp4")
        
        print(f"\nVideo Comparison Parameters:")
        print(f"  Model: {self.config.model_path}")
        print(f"  Task: {self.config.task_type.value}")
        print(f"  Video: {self.config.input_path}")
        if save_videos:
            print(f"  Output videos will be saved with _npu and _cpu suffixes")
        
        confirm = self._get_choice("\nStart full video performance comparison? (y/n)", ["y", "n", "Y", "N"], "y")
        if confirm.lower() == "y":
            self.run_video_comparison(self.config.input_path, output_path)
    
    def run_video_comparison(self, video_path: str, output_path: str = None):

        if not PERFORMANCE_FEATURES_AVAILABLE:
            print("Error: Performance comparison features not available")
            return
        
        try:

            comparison_pipeline = AdvantechVideoComparisonPipeline(
                config=self.config,
                logger=self.logger,
                video_path=video_path,
                output_path=output_path
            )
            

            comparison = comparison_pipeline.run_comparison()
            
            if comparison:
                print("\n🎯 Video performance comparison completed successfully!")
                
        except Exception as e:
            print(f"Video performance comparison failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_performance_comparison(self, model_path: str, test_frames: int = 100, warmup_frames: int = 10):

        if not self.performance_comparer:
            print("Error: Performance comparison features not available")
            return
        
        try:
            comparison = self.performance_comparer.run_comparison(
                model_path, test_frames, warmup_frames
            )
            

            if comparison:
                print(comparison.get_comparison_summary())
            
        except Exception as e:
            print(f"Performance comparison failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_dryrun(self, model_path: Optional[str] = None):
        print_banner()
        print("\n" + "="*66)
        print("  System Verification - Check Environment and Models")
        print("="*66)
        
        # Display NPU information
        if self.npu_info['available']:
            print(f"\n✅ AMD NPU: {self.npu_info['type']}")
            print(f"   Info: {self.npu_info['message']}")
            if not self.config.use_npu:
                print(f"   ⚠️  NPU will NOT be used (CPU-only mode)")
        else:
            print(f"\n⚠️  AMD NPU: Not detected")
            print(f"   Info: {self.npu_info['message']}")
        
        if model_path and Path(model_path).exists():
            print(f"\nModel: {model_path}")
            try:
                loader = AdvantechModelLoader(self.config, self.logger)
                engine = loader.load(model_path)
                print(f"   Status: Success")
                print(f"   Task Type: {engine.task_type.value}")
                print(f"   Input Size: {engine._model_input_width}x{engine._model_input_height}")
                
                # Display device type
                if hasattr(engine, 'device_type'):
                    print(f"   Device Type: {engine.device_type}")
                else:
                    print(f"   Device Type: CPU (default)")
                

                print(f"\n   Quick Performance Test (10 frames):")
                if CV2_AVAILABLE:
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    

                    for _ in range(3):
                        engine.infer(test_image)
                    

                    start_time = time.perf_counter()
                    for i in range(10):
                        engine.infer(test_image)
                    
                    total_time = time.perf_counter() - start_time
                    avg_time = total_time / 10 * 1000
                    fps = 10 / total_time if total_time > 0 else 0
                    
                    print(f"     Avg Inference Time: {avg_time:.1f}ms")
                    print(f"     Estimated FPS: {fps:.1f}")
                
                engine.cleanup()
            except Exception as e:
                print(f"   Status: Failed - {e}")
                if "ONNX" in str(e) and not ONNX_AVAILABLE:
                    print("   Hint: ONNX Runtime not installed, run: pip install onnxruntime")
                elif "torch" in str(e) and not TORCH_AVAILABLE:
                    print("   Hint: PyTorch not installed, run: pip install torch torchvision")
        else:
            print(f"\nModel: Not specified or does not exist")
        
        print("\nCameras:")
        cameras = AdvantechCameraDiscovery.discover_cameras()
        if cameras:
            for cam in cameras:
                status = "Working" if cam.get('tested', False) else "Unknown"
                print(f"  [{cam['device']}] {cam['name']} - {status}")
        else:
            print("  No cameras detected")
        
        print("\nInference Backends:")
        print(f"  ONNX Runtime: {'Available' if ONNX_AVAILABLE else 'Not available'}")
        print(f"  PyTorch: {'Available' if TORCH_AVAILABLE else 'Not available'}")
        
        print("\nPerformance Features:")
        print(f"  Performance Comparison: {'Available' if PERFORMANCE_FEATURES_AVAILABLE else 'Not available'}")
        print(f"  Video Comparison: {'Available' if PERFORMANCE_FEATURES_AVAILABLE else 'Not available'}")
        
        print("\n" + "="*66)
    
    def run_benchmark(self, model_path: str):
        print_banner()
        print("\n" + "="*66)
        print("  Benchmark Test")
        print("="*66)
        
        if not Path(model_path).exists():
            print(f"Error: Model does not exist: {model_path}")
            return
        
        print(f"\nModel: {model_path}")
        
        try:

            bench_config = AdvantechConfig()
            bench_config.model_path = model_path
            bench_config.use_npu = self.config.use_npu  
            
            model_name = Path(model_path).stem.lower()
            if '-seg' in model_name:
                bench_config.task_type = AdvantechTaskType.SEGMENTATION
            elif '-cls' in model_name:
                bench_config.task_type = AdvantechTaskType.CLASSIFICATION
            else:
                bench_config.task_type = AdvantechTaskType.DETECTION
            
            loader = AdvantechModelLoader(bench_config, self.logger)
            engine = loader.load(model_path)
            
            # Display device information
            device_type = getattr(engine, 'device_type', 'CPU')
            print(f"Device Type: {device_type}")
            if device_type == 'CPU' and self.config.use_npu == False:
                print(f"Note: Running in CPU-only mode (by user request)")
            print(f"Task Type: {engine.task_type.value}")
            
            input_h = engine._model_input_height
            input_w = engine._model_input_width
            

            test_sizes = [
                (input_w, input_h),  
                (640, 640),  
                (1280, 720),  
                (1920, 1080)  
            ]
            
            for size_idx, (width, height) in enumerate(test_sizes):
                print(f"\nTesting resolution: {width}x{height}")
                
                dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                print("Warming up...")
                warmup_latencies = []
                for _ in range(10):
                    if is_shutdown():
                        break
                    start = time.perf_counter()
                    engine.infer(dummy)
                    warmup_latencies.append((time.perf_counter() - start) * 1000)
                
                print("Running benchmark (50 iterations)...")
                latencies = []
                
                for i in range(50):
                    if is_shutdown():
                        break
                    start = time.perf_counter()
                    result = engine.infer(dummy)
                    latencies.append((time.perf_counter() - start) * 1000)
                    

                    if (i + 1) % 10 == 0:
                        print(f"  Completed {i + 1}/50 iterations")
                
                if latencies:
                    sorted_lat = sorted(latencies)
                    n = len(sorted_lat)
                    avg = sum(latencies) / n
                    
                    print(f"  Results for {width}x{height}:")
                    print(f"    Average Latency: {avg:.2f}ms")
                    print(f"    Min Latency: {min(latencies):.2f}ms")
                    print(f"    Max Latency: {max(latencies):.2f}ms")
                    print(f"    Median (P50): {sorted_lat[int(n * 0.5)]:.2f}ms")
                    print(f"    P95: {sorted_lat[int(n * 0.95)]:.2f}ms")
                    print(f"    Estimated FPS: {1000.0 / avg:.1f}")
            
            engine.cleanup()
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            if "ONNX" in str(e) and not ONNX_AVAILABLE:
                print("Hint: ONNX Runtime not installed, run: pip install onnxruntime")
            elif "torch" in str(e) and not TORCH_AVAILABLE:
                print("Hint: PyTorch not installed, run: pip install torch torchvision")
        
        print("\n" + "="*66)
    
    def _run_pipeline(self):
        if not Path(self.config.model_path).exists():
            print(f"Error: Model does not exist: {self.config.model_path}")
            return
        
        print(f"\nLoading model: {self.config.model_path}")
        
        if self.config.model_format == AdvantechModelFormat.ONNX and not ONNX_AVAILABLE:
            print("Error: ONNX Runtime not available, cannot load ONNX model")
            print("Please install: pip install onnxruntime")
            return
        
        if self.config.model_format == AdvantechModelFormat.PYTORCH and not TORCH_AVAILABLE:
            print("Error: PyTorch not available, cannot load PyTorch model")
            print("Please install: pip install torch torchvision")
            return
        
        loader = AdvantechModelLoader(self.config, self.logger)
        try:
            engine = loader.load(self.config.model_path, self.config.model_format)
            engine.warmup(iterations=self.config.warmup_iterations)
            
            # Get device type
            device_type = getattr(engine, 'device_type', 'CPU')
            print(f"  Task: {engine.task_type.value}")
            print(f"  Format: {self.config.model_format.value}")
            print(f"  Device: {device_type}")
            
            if hasattr(engine, '_model_input_width'):
                print(f"  Input Size: {engine._model_input_width}x{engine._model_input_height}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            if "ONNX" in str(e) and not ONNX_AVAILABLE:
                print("Hint: ONNX Runtime not installed, run: pip install onnxruntime")
            elif "torch" in str(e) and not TORCH_AVAILABLE:
                print("Hint: PyTorch not installed, run: pip install torch torchvision")
            import traceback
            traceback.print_exc()
            return
        
        if is_shutdown():
            engine.cleanup()
            return
        
        source = AdvantechVideoSource()
        
        if self.config.input_source == AdvantechInputSource.CAMERA:
            print(f"\nOpening camera: {self.config.camera_device}")
            try:
                if isinstance(self.config.camera_device, str):
                    if self.config.camera_device.isdigit():
                        device_num = int(self.config.camera_device)
                    elif self.config.camera_device.startswith('/dev/video'):
                        device_num = int(self.config.camera_device.split('/dev/video')[1])
                    else:
                        device_num = 0
                else:
                    device_num = int(self.config.camera_device)
            except:
                device_num = 0
            
            if not source.open_camera(device_num,
                                    width=self.config.camera_width,
                                    height=self.config.camera_height,
                                    pixel_format=self.config.camera_format,
                                    fps=self.config.camera_fps):
                engine.cleanup()
                print("Error: Cannot open camera")
                return
                
        elif self.config.input_source == AdvantechInputSource.FILE:
            print(f"\nOpening file: {self.config.input_path}")
            if not source.open_file(self.config.input_path):
                engine.cleanup()
                print("Error: Cannot open video file")
                return
        else:
            print(f"Error: This input source type is not currently supported: {self.config.input_source}")
            engine.cleanup()
            return
        
        print(f"  Resolution: {source.width}x{source.height}")
        if source.fps > 0:
            print(f"  FPS: {source.fps:.2f}")
        
        if is_shutdown():
            source.release()
            engine.cleanup()
            return
        
        self._pipeline = AdvantechPipeline(self.config, self.logger, source, engine, self.npu_info)
        
        print(f"\n{'='*80}")
        device_type = getattr(engine, 'device_type', 'CPU')
        task_name = self.config.task_type.value.upper()
        if device_type == 'AMD NPU':
            print(f"  🚀 {task_name} with NPU acceleration! Press 'q' or close window to stop.")
        else:
            print(f"  💻 {task_name} on CPU. Press 'q' or close window to stop.")
            if self.npu_info['available'] and not self.config.use_npu:
                print(f"  ⚠️  Running in CPU-only mode (by user request)")
        
        print(f"  📊 Performance monitoring: Press 'p' for details, 's' to save report")
        print("="*80 + "\n")
        
        try:
            self._pipeline.start()
        except KeyboardInterrupt:
            print("\nUser interrupted")
        except Exception as e:
            print(f"Pipeline execution exception: {e}")
        
        stats = self._pipeline.get_stats()
        print(f"\n{'='*80}")
        print(f"  Complete")
        print(f"  Task: {stats.get('task_type', 'Unknown').upper()}")
        print(f"  Device: {stats.get('device_type', 'CPU')}")
        print(f"  Processed Frames: {stats['processed_frames']}")
        print(f"  Average FPS: {stats['avg_fps']:.2f}")
        print(f"  Run Time: {stats['elapsed_time']:.2f} seconds")
        

        if 'performance_metrics' in stats:
            perf = stats['performance_metrics']
            print(f"\n  Performance Details:")
            print(f"    Avg Latency: {stats.get('latency_ms', 0):.1f}ms")
            print(f"    Memory Usage: {stats.get('memory_usage_mb', 0):.0f}MB")
            print(f"    CPU Usage: {stats.get('cpu_usage_percent', 0):.1f}%")
        
        if self.config.save_video and self.config.output_path:
            print(f"  Output: {self.config.output_path}")
        
        print("="*80 + "\n")
        
        self._pipeline.stop()
        engine.cleanup()

# ==========================================================================
# Main Function (Enhanced with Video Comparison)
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Advantech YOLO Inference Pipeline (AMD NPU + CPU with Video Performance Comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  %(prog)s --list-cameras
  %(prog)s --model yolo11n.onnx --input-source camera --camera-device 0 --task detection
  %(prog)s --model yolo11n-cls.onnx --input-source file --video-file input.mp4 --task classification
  %(prog)s --model yolo11n-seg.onnx --input-source file --video-file video.mp4 --task segmentation
  %(prog)s --benchmark --model yolo11n.onnx
  %(prog)s --dryrun --model yolo11n.onnx
  %(prog)s --model yolo11n.onnx --input-source camera --camera-device 0 --cpu-only  # Force CPU-only mode
  
  # Quick performance comparison (100 frames)
  %(prog)s --compare-performance --model yolo11n.onnx --test-frames 200
  
  # Full video performance comparison (process entire video)
  %(prog)s --model yolo11n.onnx --input-source file --video-file test.mp4 --task detection --compare-performance
  
Performance Comparison Modes:
  1. Quick comparison: --compare-performance --model model.onnx --test-frames 100
  2. Full video comparison: --compare-performance --model model.onnx --input-source file --video-file video.mp4
  
Performance Monitoring:
  During inference, press 'p' to show/hide performance details
  During inference, press 's' to save performance report
"""
    )
    
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    parser.add_argument("--dryrun", action="store_true", help="System verification")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark test")
    

    parser.add_argument("--compare-performance", action="store_true", 
                       help="Run performance comparison between NPU and CPU")
    parser.add_argument("--test-frames", type=int, default=100,
                       help="Number of frames for quick performance comparison (default: 100)")
    parser.add_argument("--warmup-frames", type=int, default=10,
                       help="Number of warmup frames before performance test (default: 10)")
    parser.add_argument("--save-report", action="store_true", default=True,
                       help="Save performance comparison report (default: True)")
    
    parser.add_argument("--model", "-m", type=str, required=False, help="Model path")
    parser.add_argument("--format", "-f", choices=["pt", "onnx"], 
                       default="onnx", help="Model format (pt: PyTorch, onnx: ONNX)")
    

    parser.add_argument("--cpu-only", action="store_true", 
                       help="Force CPU-only mode (disable NPU acceleration)")
    

    parser.add_argument("--input-source", type=str, choices=["camera", "file"], 
                       default="camera", help="Input source type (camera or file)")
    parser.add_argument("--camera-device", type=str, default="0",
                       help="Camera device number (e.g., 0, 1, /dev/video0)")
    parser.add_argument("--video-file", type=str, default="input.mp4",
                       help="Video file path")
    
    parser.add_argument("--task", "-t", choices=["detection", "classification", "segmentation"], 
                       default="detection", help="Task type")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output path")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    
    parser.add_argument("--cam-width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--cam-height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--cam-fps", type=int, default=30, help="Camera frame rate")
    parser.add_argument("--cam-format", type=str, default="MJPG", help="Camera format (MJPG, YUYV, etc.)")
    
    args = parser.parse_args()
    
    cli = AdvantechCLI()
    

    if args.cpu_only:
        cli.config.use_npu = False
        print("CPU-only mode enabled (NPU acceleration disabled)")
    
    cli.config.save_video = args.save_video
    cli.config.output_path = args.output
    cli.config.show_display = not args.no_display
    cli.config.confidence_threshold = args.conf
    cli.config.iou_threshold = args.iou
    cli.config.warmup_iterations = args.warmup
    cli.config.compare_performance = args.compare_performance
    cli.config.performance_test_frames = args.test_frames
    cli.config.save_performance_report = args.save_report
    
    task_map = {
        "detection": AdvantechTaskType.DETECTION,
        "classification": AdvantechTaskType.CLASSIFICATION,
        "segmentation": AdvantechTaskType.SEGMENTATION
    }
    cli.config.task_type = task_map[args.task]
    
    if args.format:
        format_map = {
            "pt": AdvantechModelFormat.PYTORCH,
            "onnx": AdvantechModelFormat.ONNX
        }
        cli.config.model_format = format_map.get(args.format)
    
    if args.format == "onnx" and not ONNX_AVAILABLE:
        print("[ERROR] ONNX Runtime is not available.")
        print("[INFO] Please install onnxruntime:")
        print("       pip install onnxruntime")
        sys.exit(1)
    
    if args.format == "pt" and not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is not available.")
        print("[INFO] Please install PyTorch:")
        print("       pip install torch torchvision")
        sys.exit(1)
    

    if args.list_cameras:
        cli.list_cameras()
    elif args.dryrun:
        if not args.model:
            print("Error: --dryrun requires --model")
            sys.exit(1)
        cli.run_dryrun(args.model)
    elif args.benchmark:
        if not args.model:
            print("Error: --benchmark requires --model")
            sys.exit(1)
        cli.run_benchmark(args.model)
    elif args.compare_performance:

        if not args.model:
            print("Error: --compare-performance requires --model")
            sys.exit(1)
        
        cli.config.model_path = args.model
        

        if args.input_source == "file" and args.video_file:

            print_banner()
            print("\n" + "="*80)
            print("FULL VIDEO PERFORMANCE COMPARISON MODE")
            print("="*80)
            

            cli.config.input_source = AdvantechInputSource.FILE
            cli.config.input_path = args.video_file
            

            output_path = args.output if args.save_video else None
            cli.run_video_comparison(args.video_file, output_path)
        else:

            print_banner()
            cli.run_performance_comparison(args.model, args.test_frames, args.warmup_frames)
    else:

        if not args.model:
            print("Error: Must specify model with --model")
            sys.exit(1)
        
        cli.config.model_path = args.model
        

        if args.input_source == "camera":
            cli.config.input_source = AdvantechInputSource.CAMERA
            cli.config.camera_device = args.camera_device
            cli.config.camera_width = args.cam_width
            cli.config.camera_height = args.cam_height
            cli.config.camera_format = args.cam_format
            cli.config.camera_fps = args.cam_fps
        elif args.input_source == "file":
            cli.config.input_source = AdvantechInputSource.FILE
            cli.config.input_path = args.video_file
        else:
            print(f"Error: Invalid input source: {args.input_source}")
            sys.exit(1)
        
        print_banner()
        cli._run_pipeline()

if __name__ == "__main__":
    main()

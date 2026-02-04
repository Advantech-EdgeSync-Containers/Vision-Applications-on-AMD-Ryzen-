#!/usr/bin/env python3
"""Advantech YOLO Core Module v2.1 - AMD NPU + CPU inference engine."""

__title__ = "Advantech YOLO Core Module (AMD NPU Support)"
__author__ = "Samir Singh"
__copyright__ = "Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__license__ = "Proprietary - Advantech Corporation"
__version__ = "2.1.1"  # Updated version number
__build_date__ = "2025-12-05"
__maintainer__ = "Samir Singh"

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import threading
import logging
import gc
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from collections import deque
from abc import ABC, abstractmethod

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
for attr, val in [('bool', np.bool_), ('int', np.int_), ('float', np.float64),
                  ('object', np.object_), ('str', np.str_), ('complex', np.complex128),
                  ('long', np.int_), ('unicode', np.str_)]:
    if not hasattr(np, attr):
        setattr(np, attr, val)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ==========================================================================
# Class and Configuration Definitions - Must define these base classes first
# ==========================================================================

class AdvantechTaskType(Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"

class AdvantechModelFormat(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"

class AdvantechInputSource(Enum):
    CAMERA = "camera"
    RTSP = "rtsp"
    FILE = "file"

class AdvantechDropPolicy(Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"

@dataclass
class AdvantechCameraFormat:
    pixel_format: str
    width: int
    height: int
    fps: List[int] = field(default_factory=list)

@dataclass
class AdvantechCameraInfo:
    device_id: int
    name: str
    formats: List[AdvantechCameraFormat]

@dataclass
class AdvantechConfig:
    model_path: str = ""
    model_format: Optional['AdvantechModelFormat'] = None
    task_type: AdvantechTaskType = AdvantechTaskType.DETECTION
    input_source: AdvantechInputSource = AdvantechInputSource.CAMERA
    input_path: str = ""
    camera_device: str = "0"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_format: str = "MJPG"
    rtsp_url: str = ""
    video_path: str = ""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    batch_size: int = 1
    buffer_size: int = 3
    drop_policy: AdvantechDropPolicy = AdvantechDropPolicy.DROP_OLDEST
    enable_tracking: bool = False
    show_display: bool = True
    save_video: bool = False
    output_path: str = ""
    log_level: str = "INFO"
    log_path: str = "/var/log/advantech"
    health_port: int = 8080
    warmup_iterations: int = 5
    use_npu: bool = True  # New: whether to use NPU
    
    @classmethod
    def from_env(cls) -> 'AdvantechConfig':
        config = cls()
        config.confidence_threshold = float(os.getenv('CONF_THRESHOLD', '0.25'))
        config.iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.45'))
        config.show_display = os.getenv('DISPLAY', '') != ''
        config.use_npu = os.getenv('USE_NPU', '1') == '1'
        return config

@dataclass
class AdvantechMetrics:
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    total_frames: int = 0
    dropped_frames: int = 0

@dataclass
class AdvantechDetection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = None

@dataclass
class AdvantechClassification:
    class_id: int
    class_name: str
    confidence: float
    top5: Optional[List[Tuple[int, str, float]]] = None

@dataclass
class AdvantechFrame:
    data: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int
    detections: List[AdvantechDetection] = field(default_factory=list)
    classification: Optional[AdvantechClassification] = None
    inference_time_ms: float = 0.0
    total_latency_ms: float = 0.0

# ==========================================================================
# Logger Class - Must be defined before other classes
# ==========================================================================

class AdvantechLogger:
    def __init__(self, config: AdvantechConfig):
        self.config = config
        self.logger = logging.getLogger("advantech")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
        console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
        self.logger.addHandler(console)
        
        try:
            Path(config.log_path).mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                f"{config.log_path}/advantech.log", maxBytes=10*1024*1024, backupCount=5)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            self.logger.addHandler(file_handler)
        except:
            pass
    
    def debug(self, msg: str, component: str = "Main"):
        self.logger.debug(f"[{component}] {msg}")
    
    def info(self, msg: str, component: str = "Main"):
        self.logger.info(f"[{component}] {msg}")
    
    def warning(self, msg: str, component: str = "Main"):
        self.logger.warning(f"[{component}] {msg}")
    
    def error(self, msg: str, component: str = "Main"):
        self.logger.error(f"[{component}] {msg}")

# ==========================================================================
# AMD NPU Detection and Configuration
# ==========================================================================
def detect_amd_npu_hardware() -> Dict[str, Any]:
    """Detect AMD NPU hardware"""
    npu_info = {
        'available': False,
        'type': 'UNKNOWN',
        'xclbin_path': '',
        'xclbin_required': False,
        'providers': []
    }
    
    try:
        # Check Vitis AI environment
        vitisai_env = os.environ.get('VITIS_AI_HOME', '')
        xilinx_xrt = os.environ.get('XILINX_XRT', '')
        
        # Check device files
        has_xrt_device = False
        import glob
        if len(glob.glob('/dev/xclmgmt*')) > 0 or len(glob.glob('/dev/dri/renderD*')) > 0:
            has_xrt_device = True
        
        # Check if ONNX Runtime supports Vitis AI EP
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'VitisAIExecutionProvider' in available_providers:
                npu_info['providers'].append('VitisAIExecutionProvider')
                npu_info['available'] = True
        except:
            pass
        
        # Determine NPU type
        if has_xrt_device and (vitisai_env or xilinx_xrt):
            npu_info['type'] = 'PHX/HPT'
            npu_info['xclbin_required'] = True
            
            # Find xclbin file
            xclbin_search_paths = [
                '/opt/xilinx/xrt/share/xclbin',
                '/usr/lib/xrt/share/xclbin',
                str(Path.home() / '.xclbin'),
                '/opt/vitis_ai/xclbin',
                os.path.join(vitisai_env, 'xclbin') if vitisai_env else '',
                os.path.join(xilinx_xrt, 'share/xclbin') if xilinx_xrt else ''
            ]
            
            for path in xclbin_search_paths:
                if path and os.path.exists(path):
                    for file in os.listdir(path):
                        if file.endswith('.xclbin'):
                            npu_info['xclbin_path'] = os.path.join(path, file)
                            break
                    if npu_info['xclbin_path']:
                        break
        
        elif os.path.exists('/sys/class/vai'):
            npu_info['type'] = 'STX/KRK'
            npu_info['xclbin_required'] = False
            npu_info['available'] = True
        
        # If Vitis AI EP detected but cannot determine type, mark as available
        if 'VitisAIExecutionProvider' in npu_info['providers'] and not npu_info['available']:
            npu_info['available'] = True
            npu_info['type'] = 'VitisAI-ONNX'
            
    except Exception as e:
        print(f"NPU hardware detection failed: {e}")
    
    return npu_info

def get_amd_npu_config(npu_type: str, model_path: str = "") -> Dict[str, Any]:
    """Get configuration based on NPU type"""
    config = {
        'provider': 'VitisAIExecutionProvider',
        'provider_options': {},
        'xclbin_required': False,
        'xclbin_path': ''
    }
    
    # Guess precision based on model name (simple method)
    model_name = Path(model_path).stem.lower() if model_path else ""
    is_int8 = 'int8' in model_name or 'quant' in model_name
    is_bf16 = 'bf16' in model_name or 'bfloat16' in model_name
    
    if npu_type == 'PHX/HPT':
        config['xclbin_required'] = True
        
        # Find xclbin file
        xclbin_search_paths = [
            '/opt/xilinx/xrt/share/xclbin',
            '/usr/lib/xrt/share/xclbin',
            str(Path.home() / '.xclbin'),
            '/opt/vitis_ai/xclbin'
        ]
        
        for path in xclbin_search_paths:
            if os.path.exists(path):
                # Select xclbin based on model precision
                target_file = None
                for file in os.listdir(path):
                    if file.endswith('.xclbin'):
                        if is_int8 and 'int8' in file.lower():
                            target_file = file
                            break
                        elif is_bf16 and 'bf16' in file.lower():
                            target_file = file
                            break
                        elif not is_int8 and not is_bf16 and 'default' in file.lower():
                            target_file = file
                            break
                
                if target_file:
                    config['xclbin_path'] = os.path.join(path, target_file)
                    break
        
        config['provider_options'] = {
            'cache_dir': str(Path.cwd() / '.vitisai_cache'),
            'cache_key': 'modelcache',
            'enable_cache_file_io_in_mem': '0',
            'target': 'X1',
            'xclbin': config['xclbin_path'] if config['xclbin_path'] else ''
        }
    
    elif npu_type in ['STX', 'KRK']:
        config['xclbin_required'] = False
        config['provider_options'] = {
            'cache_dir': str(Path.cwd() / '.vitisai_cache'),
            'cache_key': 'modelcache',
            'enable_cache_file_io_in_mem': '0'
        }
    
    else:  # Generic Vitis AI configuration
        config['provider_options'] = {
            'cache_dir': str(Path.cwd() / '.vitisai_cache'),
            'cache_key': 'modelcache',
            'enable_cache_file_io_in_mem': '0'
        }
    
    return config

# ==========================================================================
# ONNX Runtime Import and Setup
# ==========================================================================
ONNX_AVAILABLE = False
ort = None
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

# ==========================================================================
# Class Data - Simplified version to avoid external file dependencies
# ==========================================================================

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Simplified ImageNet classes (first 100, for testing)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch",
    "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay",
    "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle",
    "vulture", "great grey owl", "fire salamander", "smooth newt", "newt",
    "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog",
    "loggerhead turtle", "leatherback turtle", "mud turtle", "terrapin", "box turtle",
    "banded gecko", "common iguana", "American chameleon", "whiptail lizard",
    "agama", "frilled lizard", "alligator lizard", "Gila monster", "green lizard",
    "African chameleon", "Komodo dragon", "African crocodile", "American alligator",
    "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake",
    "king snake", "garter snake", "water snake", "vine snake", "night snake",
    "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake",
    "horned viper", "diamondback rattlesnake", "sidewinder", "trilobite", "harvestman",
    "scorpion", "black and gold garden spider", "barn spider", "garden spider",
    "black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse",
    "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge",
    "African grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan",
    "drake", "red-breasted merganser", "goose", "black swan", "tusker"
]

def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Generate a color for a class ID."""
    import random
    random.seed(class_id * 12345)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    return get_color_for_class(class_id)

def get_class_name(class_id: int, task: str = "detection") -> str:
    classes = IMAGENET_CLASSES if task == "classification" else COCO_CLASSES
    return classes[class_id] if class_id < len(classes) else f"class_{class_id}"

def get_class_names(task: str = "detection") -> List[str]:
    return IMAGENET_CLASSES if task == "classification" else COCO_CLASSES

# ==========================================================================
# Other Support Classes
# ==========================================================================

class AdvantechMemoryManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._buffers: Dict[str, Any] = {}
        self._initialized = True
    
    def allocate(self, name: str, shape: Tuple, dtype: np.dtype = np.float32) -> np.ndarray:
        key = f"{name}_{shape}_{dtype}"
        if key not in self._buffers:
            self._buffers[key] = np.empty(shape, dtype=dtype)
        return self._buffers[key]
    
    def cleanup(self):
        self._buffers.clear()
        gc.collect()

class AdvantechMetricsCollector:
    def __init__(self, window_size: int = 100):
        self._latencies = deque(maxlen=window_size)
        self._timestamps = deque(maxlen=window_size)
        self._total_frames = 0
        self._dropped_frames = 0
        self._lock = threading.Lock()
    
    def record_frame(self):
        with self._lock:
            self._timestamps.append(time.perf_counter())
            self._total_frames += 1
    
    def record_latency(self, latency_ms: float):
        with self._lock:
            self._latencies.append(latency_ms)
    
    def record_drop(self):
        with self._lock:
            self._dropped_frames += 1
    
    def get_metrics(self) -> AdvantechMetrics:
        with self._lock:
            fps = 0.0
            if len(self._timestamps) >= 2:
                elapsed = self._timestamps[-1] - self._timestamps[0]
                if elapsed > 0:
                    fps = (len(self._timestamps) - 1) / elapsed
            
            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0
            max_lat = max(self._latencies) if self._latencies else 0
            min_lat = min(self._latencies) if self._latencies else float('inf')
            
            return AdvantechMetrics(
                fps=fps, avg_latency_ms=avg_lat, max_latency_ms=max_lat,
                min_latency_ms=min_lat, total_frames=self._total_frames,
                dropped_frames=self._dropped_frames)
    
    def get_total_fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
    
    def get_total_frames(self) -> int:
        return self._total_frames

class AdvantechRingBuffer:
    def __init__(self, capacity: int, drop_policy: AdvantechDropPolicy = AdvantechDropPolicy.DROP_OLDEST):
        self._buffer = deque(maxlen=capacity)
        self._drop_policy = drop_policy
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped = 0
    
    def put(self, item: Any) -> bool:
        with self._lock:
            if len(self._buffer) >= self._buffer.maxlen:
                if self._drop_policy == AdvantechDropPolicy.DROP_OLDEST:
                    self._buffer.popleft()
                    self._dropped += 1
                elif self._drop_policy == AdvantechDropPolicy.DROP_NEWEST:
                    self._dropped += 1
                    return False
            self._buffer.append(item)
            self._not_empty.notify()
            return True
    
    def get(self, timeout: float = None) -> Optional[Any]:
        with self._not_empty:
            if not self._buffer:
                self._not_empty.wait(timeout)
            if self._buffer:
                return self._buffer.popleft()
            return None
    
    def clear(self):
        with self._lock:
            self._buffer.clear()
    
    @property
    def dropped_count(self) -> int:
        return self._dropped

# ==========================================================================
# Core Engine Class - Supports AMD NPU (fixed providers error)
# ==========================================================================

class AdvantechEngine(ABC):
    def __init__(self, model_path: str, config: AdvantechConfig, logger: AdvantechLogger):
        self.model_path = model_path
        self.config = config
        self.logger = logger
        self.task_type = config.task_type
        self._model_input_width = 640
        self._model_input_height = 640
        self._last_scale = 1.0
        self._last_pad_x = 0
        self._last_pad_y = 0
        self.device_type = "CPU"  # Default device type
    
    @abstractmethod
    def infer(self, frame: np.ndarray) -> Union[List[AdvantechDetection], AdvantechClassification]:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
    
    def warmup(self, iterations: int = 5):
        dummy = np.random.randint(0, 255, (self._model_input_height, self._model_input_width, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)
    
    def get_class_names(self) -> List[str]:
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            return IMAGENET_CLASSES
        return COCO_CLASSES
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_h, target_w = self._model_input_height, self._model_input_width
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        self._last_scale = scale
        self._last_pad_x = pad_x
        self._last_pad_y = pad_y
        
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(blob)
    
    def preprocess_classification(self, frame: np.ndarray) -> np.ndarray:
        target_h, target_w = self._model_input_height, self._model_input_width
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(blob)
    
    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int],
                   num_masks: int = 0) -> List[AdvantechDetection]:
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        num_cols = output.shape[1]
        num_classes = num_cols - 4 - num_masks
        if num_classes <= 0:
            return []
        
        boxes = output[:, :4]
        class_scores = output[:, 4:4 + num_classes]
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        valid_mask = confidences >= self.config.confidence_threshold
        if not np.any(valid_mask):
            return []
        
        boxes = boxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]
        
        h_orig, w_orig = original_shape
        scale = self._last_scale
        pad_x, pad_y = self._last_pad_x, self._last_pad_y
        
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = np.clip((boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale, 0, w_orig)
        boxes_xyxy[:, 1] = np.clip((boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale, 0, h_orig)
        boxes_xyxy[:, 2] = np.clip((boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale, 0, w_orig)
        boxes_xyxy[:, 3] = np.clip((boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale, 0, h_orig)
        
        indices = self._nms(boxes_xyxy, confidences, self.config.iou_threshold)[:self.config.max_detections]
        class_names = self.get_class_names()
        
        return [AdvantechDetection(
            bbox=tuple(boxes_xyxy[i]),
            confidence=float(confidences[i]),
            class_id=int(class_ids[i]),
            class_name=class_names[class_ids[i]] if class_ids[i] < len(class_names) else f"class_{class_ids[i]}"
        ) for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            order = order[1:][iou <= iou_threshold]
        
        return keep

class AdvantechOnnxEngine(AdvantechEngine):
    def __init__(self, model_path: str, config: AdvantechConfig, logger: AdvantechLogger):
        super().__init__(model_path, config, logger)
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        
        # Detect AMD NPU
        npu_hardware_info = detect_amd_npu_hardware()
        self.npu_available = npu_hardware_info['available'] and config.use_npu
        self.npu_type = npu_hardware_info['type']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 0
        
        # Fix: Ensure providers and provider_options have the same length
        providers = []
        provider_options = []
        
        # Try to use AMD NPU
        if self.npu_available and 'VitisAIExecutionProvider' in npu_hardware_info['providers']:
            try:
                npu_config = get_amd_npu_config(self.npu_type, model_path)
                
                if self.npu_type == 'PHX/HPT' and npu_config['xclbin_required']:
                    if not npu_config['xclbin_path'] or not os.path.exists(npu_config['xclbin_path']):
                        self.logger.warning(f"xclbin file not found, NPU acceleration may not be available")
                        # Continue trying, some NPUs might not need xclbin
                
                providers.append('VitisAIExecutionProvider')
                provider_options.append(npu_config['provider_options'])
                
                self.logger.info(f"Trying to use AMD NPU acceleration ({self.npu_type})")
                if npu_config['xclbin_path']:
                    self.logger.info(f"Using xclbin: {npu_config['xclbin_path']}")
                
            except Exception as e:
                self.logger.warning(f"NPU configuration failed: {e}")
                self.npu_available = False
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        provider_options.append({})  # CPU provider uses empty config dictionary
        
        self.logger.info(f"Execution providers: {providers}")
        self.logger.info(f"Provider options: {provider_options}")
        
        # Try to create session
        try:
            # Create inference session - providers and provider_options must have same length
            self._session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options, 
                providers=providers,
                provider_options=provider_options
            )
            
            # Determine actual provider being used
            actual_provider = self._session.get_providers()[0]
            if 'VitisAI' in actual_provider and self.npu_available:
                self.device_type = "AMD NPU"
                self.logger.info(f"✅ Successfully enabled AMD NPU acceleration ({self.npu_type})")
            else:
                self.device_type = "CPU"
                if self.npu_available:
                    self.logger.warning(f"⚠️ NPU available but actually using CPU, check model compatibility or configuration")
                else:
                    self.logger.info(f"💻 Using CPU inference")
            
            # Get model input/output information
            input_info = self._session.get_inputs()[0]
            shape = input_info.shape
            self._input_name = input_info.name
            self._input_dtype = input_info.type
            self._use_fp16 = 'float16' in self._input_dtype or 'Float16' in self._input_dtype
            
            if len(shape) >= 4:
                self._model_input_height = shape[2] if isinstance(shape[2], int) else 640
                self._model_input_width = shape[3] if isinstance(shape[3], int) else 640
            else:
                self._model_input_height = 640
                self._model_input_width = 640
                
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._identify_outputs()
            
            self.logger.info(f"ONNX model loaded successfully: {model_path}")
            self.logger.info(f"Device: {self.device_type}")
            self.logger.info(f"Input: {self._input_name}, Output: {self._output_names}")
            self.logger.info(f"Input size: {self._model_input_width}x{self._model_input_height}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            
            # If NPU fails, try pure CPU
            if self.npu_available:
                self.logger.info("NPU failed, trying pure CPU mode...")
                try:
                    self._session = ort.InferenceSession(
                        model_path, 
                        sess_options=sess_options, 
                        providers=['CPUExecutionProvider']
                    )
                    self.device_type = "CPU"
                    self.logger.info("✅ Pure CPU mode successful")
                    
                    input_info = self._session.get_inputs()[0]
                    shape = input_info.shape
                    self._input_name = input_info.name
                    
                    if len(shape) >= 4:
                        self._model_input_height = shape[2] if isinstance(shape[2], int) else 640
                        self._model_input_width = shape[3] if isinstance(shape[3], int) else 640
                    
                    self._output_names = [o.name for o in self._session.get_outputs()]
                    self._identify_outputs()
                    
                except Exception as e2:
                    raise RuntimeError(f"CPU fallback also failed: {e2}")
            else:
                raise

    def _identify_outputs(self):
        outputs = self._session.get_outputs()
        for idx, out in enumerate(outputs):
            shape = out.shape
            if len(shape) == 4 and (shape[1] == 32 or (isinstance(shape[1], int) and shape[1] == 32)):
                self._mask_output_idx = idx
                self.task_type = AdvantechTaskType.SEGMENTATION
                return
        if len(outputs) == 1:
            out_shape = outputs[0].shape
            if len(out_shape) == 2 and isinstance(out_shape[1], int) and out_shape[1] >= 100:
                if self._model_input_height <= 256 or '-cls' in self.model_path.lower() or '_cls' in self.model_path.lower():
                    self.task_type = AdvantechTaskType.CLASSIFICATION

    def infer(self, frame: np.ndarray) -> Union[List[AdvantechDetection], AdvantechClassification]:
        original_shape = (frame.shape[0], frame.shape[1])
        
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            blob = self.preprocess_classification(frame)
        else:
            blob = self.preprocess(frame)
        
        if self._use_fp16:
            blob = blob.astype(np.float16)
        else:
            blob = blob.astype(np.float32)
        
        try:
            outputs = self._session.run(self._output_names, {self._input_name: blob})
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return [] if self.task_type != AdvantechTaskType.CLASSIFICATION else None
        
        if self._use_fp16:
            outputs = [o.astype(np.float32) if o.dtype == np.float16 else o for o in outputs]
        
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            return self.postprocess_classification(outputs[0])
        elif self.task_type == AdvantechTaskType.SEGMENTATION and hasattr(self, '_mask_output_idx') and self._mask_output_idx >= 0:
            det_idx = 1 if self._mask_output_idx == 0 else 0
            return self.postprocess_segmentation(outputs[det_idx], outputs[self._mask_output_idx], original_shape)
        
        return self.postprocess(outputs[0], original_shape)
    
    def postprocess_segmentation(self, det_output: np.ndarray, mask_protos: np.ndarray,
                                 original_shape: Tuple[int, int]) -> List[AdvantechDetection]:
        det_output = det_output.astype(np.float32)
        mask_protos = mask_protos.astype(np.float32)
        
        if det_output.ndim == 3:
            det_output = det_output[0]
        if det_output.shape[0] < det_output.shape[1]:
            det_output = det_output.T
        
        num_cols = det_output.shape[1]
        num_mask_coeffs = 32
        num_classes = num_cols - 4 - num_mask_coeffs
        
        if num_classes <= 0:
            return self.postprocess(det_output, original_shape, num_masks=0)
        
        boxes = det_output[:, :4]
        class_scores = det_output[:, 4:4 + num_classes]
        mask_coeffs = det_output[:, 4 + num_classes:4 + num_classes + num_mask_coeffs]
        
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        valid_mask = confidences >= self.config.confidence_threshold
        
        if not np.any(valid_mask):
            return []
        
        boxes = boxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]
        mask_coeffs = mask_coeffs[valid_mask]
        
        h_orig, w_orig = original_shape
        target_h, target_w = self._model_input_height, self._model_input_height
        scale = self._last_scale
        pad_x, pad_y = self._last_pad_x, self._last_pad_y
        
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = np.clip((boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale, 0, w_orig)
        boxes_xyxy[:, 1] = np.clip((boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale, 0, h_orig)
        boxes_xyxy[:, 2] = np.clip((boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale, 0, w_orig)
        boxes_xyxy[:, 3] = np.clip((boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale, 0, h_orig)
        
        indices = self._nms(boxes_xyxy, confidences, self.config.iou_threshold)[:self.config.max_detections]
        protos = mask_protos[0] if mask_protos.ndim == 4 else mask_protos
        class_names = self.get_class_names()
        detections = []
        
        for idx in indices:
            class_id = int(class_ids[idx])
            x1, y1, x2, y2 = boxes_xyxy[idx]
            
            det = AdvantechDetection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(confidences[idx]),
                class_id=class_id,
                class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            )
            
            try:
                coeffs = mask_coeffs[idx]
                mask_pred = 1.0 / (1.0 + np.exp(-np.clip(np.tensordot(coeffs, protos, axes=([0], [0])), -50, 50)))
                mask_model = cv2.resize(mask_pred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                scaled_h, scaled_w = int(h_orig * scale), int(w_orig * scale)
                y_start, y_end = max(0, pad_y), min(target_h, pad_y + scaled_h)
                x_start, x_end = max(0, pad_x), min(target_w, pad_x + scaled_w)
                
                if y_end > y_start and x_end > x_start:
                    mask_cropped = mask_model[y_start:y_end, x_start:x_end]
                    mask_orig = cv2.resize(mask_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                else:
                    mask_orig = cv2.resize(mask_pred, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
                ix1, iy1 = int(max(0, x1)), int(max(0, y1))
                ix2, iy2 = int(min(w_orig, x2)), int(min(h_orig, y2))
                
                full_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
                if ix2 > ix1 and iy2 > iy1:
                    full_mask[iy1:iy2, ix1:ix2] = (mask_orig[iy1:iy2, ix1:ix2] > 0.5).astype(np.float32)
                det.mask = full_mask
            except:
                pass
            
            detections.append(det)
        return detections
    
    def postprocess_classification(self, output: np.ndarray) -> AdvantechClassification:
        if output.ndim > 1:
            output = output.flatten()
        probs = np.exp(output - np.max(output))
        probs = probs / probs.sum()
        top5_idx = np.argsort(probs)[-5:][::-1]
        class_names = self.get_class_names()
        num_classes = len(class_names)
        top_idx = int(top5_idx[0])
        if top_idx < num_classes:
            top_name = class_names[top_idx]
        else:
            top_name = f"class_{top_idx}"
        return AdvantechClassification(
            class_id=top_idx,
            class_name=top_name,
            confidence=float(probs[top_idx]),
            top5=[(int(i), class_names[i] if i < num_classes else f"class_{i}", float(probs[i])) for i in top5_idx]
        )
    
    def cleanup(self):
        self._session = None

# ==========================================================================
# PyTorch Engine (CPU only)
# ==========================================================================

class AdvantechPyTorchEngine(AdvantechEngine):
    def __init__(self, model_path: str, config: AdvantechConfig, logger: AdvantechLogger):
        super().__init__(model_path, config, logger)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        self.device_type = "CPU"
        
        try:
            from ultralytics import YOLO
            self._device = 'cpu'
            self._model = YOLO(model_path)
            self._model.to(self._device)
            self.logger.info(f"PyTorch model loaded successfully: {model_path}")
            self.logger.info(f"Device: {self.device_type}")
        except ImportError:
            raise RuntimeError("Ultralytics not available")
    
    def infer(self, frame: np.ndarray) -> Union[List[AdvantechDetection], AdvantechClassification]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self._model(frame_rgb, conf=self.config.confidence_threshold,
                             iou=self.config.iou_threshold, verbose=False, device=self._device)
        
        if not results:
            return []
        
        result = results[0]
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            probs = result.probs
            if probs is not None:
                if hasattr(probs, 'data'):
                    probs_tensor = probs.data
                else:
                    probs_tensor = probs
                if hasattr(probs_tensor, 'cpu'):
                    probs_data = probs_tensor.cpu().numpy()
                else:
                    probs_data = np.array(probs_tensor)
                if probs_data.ndim > 1:
                    probs_data = probs_data.flatten()
                top_idx = int(np.argmax(probs_data))
                top_conf = float(probs_data[top_idx])
                top5_indices = np.argsort(probs_data)[-5:][::-1].tolist()
                class_name = IMAGENET_CLASSES[top_idx] if top_idx < len(IMAGENET_CLASSES) else f"class_{top_idx}"
                return AdvantechClassification(
                    class_id=top_idx,
                    class_name=class_name,
                    confidence=top_conf,
                    top5=[(int(i), IMAGENET_CLASSES[i] if i < len(IMAGENET_CLASSES) else f"class_{i}", float(probs_data[i]))
                          for i in top5_indices]
                )
        
        class_names = self.get_class_names()
        boxes = result.boxes
        if boxes is None:
            return []
        
        detections = []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            det = AdvantechDetection(
                bbox=tuple(box.xyxy[0].tolist()),
                confidence=float(box.conf[0]),
                class_id=class_id,
                class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                mask=masks[i] if masks is not None else None
            )
            detections.append(det)
        
        return detections
    
    def cleanup(self):
        self._model = None
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

# ==========================================================================
# Model Loader
# ==========================================================================

class AdvantechModelLoader:
    def __init__(self, config: AdvantechConfig, logger: AdvantechLogger):
        self.config = config
        self.logger = logger
    
    def detect_format(self, model_path: str) -> AdvantechModelFormat:
        suffix = Path(model_path).suffix.lower()
        if suffix in [".pt", ".pth"]:
            return AdvantechModelFormat.PYTORCH
        elif suffix == ".onnx":
            return AdvantechModelFormat.ONNX
        raise ValueError(f"Unsupported model format: {suffix}")
    
    def load(self, model_path: str, model_format: Optional[AdvantechModelFormat] = None) -> AdvantechEngine:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if model_format is None:
            model_format = self.detect_format(model_path)
        
        model_name = Path(model_path).stem.lower()
        if '-seg' in model_name or '_seg' in model_name:
            self.config.task_type = AdvantechTaskType.SEGMENTATION
        elif '-cls' in model_name or '_cls' in model_name:
            self.config.task_type = AdvantechTaskType.CLASSIFICATION
        
        if model_format == AdvantechModelFormat.ONNX:
            return AdvantechOnnxEngine(model_path, self.config, self.logger)
        elif model_format == AdvantechModelFormat.PYTORCH:
            return AdvantechPyTorchEngine(model_path, self.config, self.logger)
        
        raise ValueError(f"Unsupported format: {model_format}")

# ==========================================================================
# Renderer
# ==========================================================================

class AdvantechOverlayRenderer:
    def __init__(self, config: AdvantechConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, detections: List[AdvantechDetection],
               metrics: AdvantechMetrics, classification: Optional[AdvantechClassification] = None) -> np.ndarray:
        output = frame.copy()
        
        if self.config.task_type == AdvantechTaskType.SEGMENTATION:
            for det in detections:
                if det.mask is not None and np.any(det.mask > 0):
                    color = get_class_color(det.class_id)
                    mask_resized = cv2.resize(det.mask.astype(np.uint8), (output.shape[1], output.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    overlay = output.copy()
                    overlay[mask_resized > 0] = color
                    output = cv2.addWeighted(output, 0.6, overlay, 0.4, 0)
        
        if self.config.task_type != AdvantechTaskType.CLASSIFICATION:
            for det in detections:
                color = get_class_color(det.class_id)
                x1, y1, x2, y2 = map(int, det.bbox)
                
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det.class_name}: {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
                cv2.rectangle(output, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(output, label, (x1, y1 - 5), self.font, 0.6, (255, 255, 255), 2)
        
        if classification:
            label = f"{classification.class_name} ({classification.confidence*100:.1f}%)"
            cv2.putText(output, label, (10, 60), self.font, 1.0, (0, 255, 0), 2)
            
            if hasattr(classification, 'top5') and classification.top5:
                y_pos = 100
                for i, (cls_id, cls_name, conf) in enumerate(classification.top5[:5]):
                    text = f"{i+1}. {cls_name}: {conf*100:.1f}%"
                    cv2.putText(output, text, (10, y_pos), self.font, 0.5, (255, 255, 255), 1)
                    y_pos += 25
        
        cv2.putText(output, f"FPS: {metrics.fps:.1f}", (10, 30), self.font, 0.8, (0, 255, 0), 2)
        
        return output
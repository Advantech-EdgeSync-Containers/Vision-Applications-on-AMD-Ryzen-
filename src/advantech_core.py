#!/usr/bin/env python3
"""Advantech YOLO Core Module v2.2 - AMD NPU + CPU inference engine with performance comparison."""

__title__ = "Advantech YOLO Core Module (AMD NPU Support with Performance Comparison)"
__author__ = "Samir Singh"
__copyright__ = "Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__license__ = "Proprietary - Advantech Corporation"
__version__ = "2.2.0"
__build_date__ = "2025-12-10"
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
import json
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from datetime import datetime

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
for attr, val in [('bool', np.bool_), ('int', np.int_), ('float', np.float64),
                  ('object', np.object_), ('str', np.str_), ('complex', np.complex128),
                  ('long', np.int_), ('unicode', np.str_)]:
    if not hasattr(np, attr):
        setattr(np, attr, val)

# CV2 availability check
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None

# Torch availability check
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None


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
    use_npu: bool = True
    custom_classes_file: str = ""
    compare_performance: bool = False
    performance_test_frames: int = 100
    save_performance_report: bool = True
    
    @classmethod
    def from_env(cls) -> 'AdvantechConfig':
        config = cls()
        config.confidence_threshold = float(os.getenv('CONF_THRESHOLD', '0.25'))
        config.iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.45'))
        config.show_display = os.getenv('DISPLAY', '') != ''
        config.use_npu = os.getenv('USE_NPU', '1') == '1'
        config.custom_classes_file = os.getenv('CUSTOM_CLASSES_FILE', '')
        config.compare_performance = os.getenv('COMPARE_PERFORMANCE', '0') == '1'
        config.performance_test_frames = int(os.getenv('PERFORMANCE_TEST_FRAMES', '100'))
        return config

@dataclass
class AdvantechMetrics:
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    total_frames: int = 0
    dropped_frames: int = 0

# ==========================================================================
# 定义 Logger 类（在其他类之前）
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


try:
    from advantech_classes import (
        COCO_CLASSES,
        IMAGENET_CLASSES,
        get_color_for_class,
        get_class_name,
        get_class_names,
        load_custom_classes,
        CLASS_MANAGER
    )
    print("✅ Successfully loaded class definitions from advantech_classes.py")
except ImportError as e:
    print(f"⚠️ Warning: Cannot import advantech_classes module: {e}")
    print("⚠️ Falling back to embedded class definitions")
    
    # Fallback: Minimal class definitions
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
    
    # Minimal ImageNet classes (first 100)
    IMAGENET_CLASSES = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
        "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch",
        "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay",
        "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle",
        "vulture", "great grey owl", "European fire salamander", "common newt", "eft",
        "spotted salamander", "axolotl", "bullfrog", "tree frog", "tailed frog",
        "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
        "box turtle", "banded gecko", "common iguana", "American chameleon", "whiptail lizard",
        "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "green lizard",
        "African chameleon", "Komodo dragon", "African crocodile", "American alligator",
        "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake",
        "king snake", "garter snake", "water snake", "vine snake", "night snake",
        "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake",
        "horned viper", "diamondback rattlesnake", "sidewinder rattlesnake", "trilobite",
        "harvestman", "scorpion", "black and gold garden spider", "barn spider",
        "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede",
        "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock",
        "quail", "partridge", "African grey parrot", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar",
        "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker"
    ]
    
    def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
        """Generate a color for a class ID."""
        import random
        random.seed(class_id * 12345)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return color
    
    def get_class_name(class_id: int, task: str = "detection") -> str:
        """Get class name for a given class ID and task type."""
        classes = IMAGENET_CLASSES if task == "classification" else COCO_CLASSES
        
        if class_id < 0:
            return f"invalid_class_{class_id}"
        
        if class_id < len(classes):
            class_name = classes[class_id]
            if not class_name or class_name.strip() == "":
                return f"unknown_class_{class_id}"
            return class_name
        else:
            if task == "classification":
                if class_id >= 1000 and class_id < 2000:
                    return f"imagenet21k_class_{class_id}"
                else:
                    return f"unknown_class_{class_id}"
            else:
                return f"class_{class_id}"
    
    def get_class_names(task: str = "detection") -> List[str]:
        """Get the entire class list for a given task type."""
        return IMAGENET_CLASSES if task == "classification" else COCO_CLASSES


    class SimpleClassManager:
        def __init__(self):
            self._detection_classes = COCO_CLASSES
            self._classification_classes = IMAGENET_CLASSES
            self._custom_classes = {}
        
        def set_custom_classes(self, task: str, classes: List[str]):
            if task == "detection":
                self._detection_classes = classes
            elif task == "classification":
                self._classification_classes = classes
        
        def get_classes(self, task: str) -> List[str]:
            if task == "classification":
                return self._classification_classes
            else:
                return self._detection_classes
        
        def get_class_name(self, class_id: int, task: str = "detection") -> str:
            return get_class_name(class_id, task)
    
    CLASS_MANAGER = SimpleClassManager()
    
    def load_custom_classes(file_path: str, task: str = "detection") -> List[str]:

        try:
            with open(file_path, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(classes)} custom classes from {file_path}")
            return classes
        except Exception as e:
            print(f"Failed to load custom classes from {file_path}: {e}")
            return get_class_names(task)


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


@dataclass
class AdvantechPerformanceMetrics:

    device_type: str = "CPU"
    model_name: str = ""
    total_frames: int = 0
    total_time_ms: float = 0.0
    avg_inference_time_ms: float = 0.0
    avg_preprocess_time_ms: float = 0.0
    avg_postprocess_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    min_inference_time_ms: float = float('inf')
    fps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0  
    power_consumption_w: float = 0.0  
    throughput_fps: float = 0.0
    latency_ms: float = 0.0
    energy_efficiency_mj_per_frame: float = 0.0  
    

    inference_times: List[float] = field(default_factory=list)
    preprocess_times: List[float] = field(default_factory=list)
    postprocess_times: List[float] = field(default_factory=list)
    
    def update(self, inference_time_ms: float, preprocess_time_ms: float = 0, 
               postprocess_time_ms: float = 0, memory_mb: float = 0, cpu_percent: float = 0):

        self.total_frames += 1
        self.total_time_ms += inference_time_ms + preprocess_time_ms + postprocess_time_ms
        

        self.inference_times.append(inference_time_ms)
        self.preprocess_times.append(preprocess_time_ms)
        self.postprocess_times.append(postprocess_time_ms)
        

        self.avg_inference_time_ms = sum(self.inference_times) / len(self.inference_times)
        if self.preprocess_times:
            self.avg_preprocess_time_ms = sum(self.preprocess_times) / len(self.preprocess_times)
        if self.postprocess_times:
            self.avg_postprocess_time_ms = sum(self.postprocess_times) / len(self.postprocess_times)
        

        total_times = [inf + pre + post for inf, pre, post in 
                      zip(self.inference_times, self.preprocess_times, self.postprocess_times)]
        self.avg_total_time_ms = sum(total_times) / len(total_times) if total_times else 0
        

        self.max_inference_time_ms = max(self.max_inference_time_ms, inference_time_ms)
        self.min_inference_time_ms = min(self.min_inference_time_ms, inference_time_ms)
        

        if self.avg_total_time_ms > 0:
            self.fps = 1000.0 / self.avg_total_time_ms
            self.throughput_fps = self.fps
        

        self.latency_ms = self.avg_total_time_ms
        

        self.memory_usage_mb = memory_mb
        self.cpu_usage_percent = cpu_percent
        

        if self.power_consumption_w > 0 and self.fps > 0:
            self.energy_efficiency_mj_per_frame = (self.power_consumption_w * 1000) / self.fps
    
    def to_dict(self) -> Dict[str, Any]:

        return {
            'device_type': self.device_type,
            'model_name': self.model_name,
            'total_frames': self.total_frames,
            'total_time_ms': self.total_time_ms,
            'avg_inference_time_ms': self.avg_inference_time_ms,
            'avg_preprocess_time_ms': self.avg_preprocess_time_ms,
            'avg_postprocess_time_ms': self.avg_postprocess_time_ms,
            'avg_total_time_ms': self.avg_total_time_ms,
            'max_inference_time_ms': self.max_inference_time_ms,
            'min_inference_time_ms': self.min_inference_time_ms,
            'fps': self.fps,
            'throughput_fps': self.throughput_fps,
            'latency_ms': self.latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'gpu_usage_percent': self.gpu_usage_percent,
            'power_consumption_w': self.power_consumption_w,
            'energy_efficiency_mj_per_frame': self.energy_efficiency_mj_per_frame
        }
    
    def get_summary(self) -> str:

        summary = (f"Device: {self.device_type}\n"
                  f"  Model: {self.model_name}\n"
                  f"  FPS: {self.fps:.2f}\n"
                  f"  Throughput: {self.throughput_fps:.2f} FPS\n"
                  f"  Latency: {self.latency_ms:.2f}ms\n"
                  f"  Avg Total Time: {self.avg_total_time_ms:.2f}ms\n"
                  f"    - Inference: {self.avg_inference_time_ms:.2f}ms\n"
                  f"    - Preprocess: {self.avg_preprocess_time_ms:.2f}ms\n"
                  f"    - Postprocess: {self.avg_postprocess_time_ms:.2f}ms\n"
                  f"  Min/Max Inference: {self.min_inference_time_ms:.2f}/{self.max_inference_time_ms:.2f}ms\n"
                  f"  Memory Usage: {self.memory_usage_mb:.2f} MB\n"
                  f"  CPU Usage: {self.cpu_usage_percent:.1f}%\n")
        
        if self.power_consumption_w > 0:
            summary += f"  Power: {self.power_consumption_w:.1f}W\n"
            summary += f"  Energy Efficiency: {self.energy_efficiency_mj_per_frame:.2f} mJ/frame\n"
        
        summary += f"  Total Frames: {self.total_frames}"
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:

        if not self.inference_times:
            return {}
        
        sorted_times = sorted(self.inference_times)
        n = len(sorted_times)
        
        return {
            'p50': sorted_times[int(n * 0.5)] if n > 0 else 0,
            'p95': sorted_times[int(n * 0.95)] if n > 0 else 0,
            'p99': sorted_times[int(n * 0.99)] if n > 0 else 0,
            'std_dev': np.std(self.inference_times) if n > 1 else 0,
            'variance': np.var(self.inference_times) if n > 1 else 0
        }

@dataclass
class AdvantechPerformanceComparison:

    npu_metrics: AdvantechPerformanceMetrics
    cpu_metrics: AdvantechPerformanceMetrics
    comparison_timestamp: str = ""
    test_duration_seconds: float = 0.0
    speedup_ratio: float = 0.0
    efficiency_gain: float = 0.0
    energy_savings_percent: float = 0.0
    recommendation: str = ""
    
    def __post_init__(self):
        if not self.comparison_timestamp:
            self.comparison_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        

        if self.cpu_metrics.fps > 0 and self.npu_metrics.fps > 0:
            self.speedup_ratio = self.npu_metrics.fps / self.cpu_metrics.fps
            

            if self.cpu_metrics.energy_efficiency_mj_per_frame > 0 and self.npu_metrics.energy_efficiency_mj_per_frame > 0:
                self.efficiency_gain = self.cpu_metrics.energy_efficiency_mj_per_frame / self.npu_metrics.energy_efficiency_mj_per_frame
                self.energy_savings_percent = (1 - self.npu_metrics.energy_efficiency_mj_per_frame / self.cpu_metrics.energy_efficiency_mj_per_frame) * 100
        

        if self.speedup_ratio >= 2.0:
            self.recommendation = "✅ STRONGLY RECOMMEND using NPU - Significant performance improvement"
        elif self.speedup_ratio >= 1.2:
            self.recommendation = "👍 RECOMMEND using NPU - Good performance improvement"
        elif self.speedup_ratio >= 0.8:
            self.recommendation = "🤔 CONSIDER using NPU - Similar performance, check energy efficiency"
        else:
            self.recommendation = "⚠️ CONSIDER using CPU - NPU may be slower"
    
    def get_comparison_summary(self) -> str:

        summary = f"\n{'='*80}\n"
        summary += "ADVANTECH YOLO PERFORMANCE COMPARISON REPORT\n"
        summary += f"{'='*80}\n"
        summary += f"Test Time: {self.comparison_timestamp}\n"
        summary += f"Duration: {self.test_duration_seconds:.2f} seconds\n"
        summary += f"\n{'='*80}\n"
        

        summary += "AMD NPU PERFORMANCE:\n"
        summary += "-" * 40 + "\n"
        summary += self.npu_metrics.get_summary()
        

        summary += f"\n\n{'='*80}\n"
        summary += "CPU PERFORMANCE:\n"
        summary += "-" * 40 + "\n"
        summary += self.cpu_metrics.get_summary()
        

        summary += f"\n\n{'='*80}\n"
        summary += "PERFORMANCE COMPARISON:\n"
        summary += "-" * 40 + "\n"
        summary += f"Speedup Ratio (NPU vs CPU): {self.speedup_ratio:.2f}x\n"
        
        if self.speedup_ratio > 1:
            summary += f"  → NPU is {self.speedup_ratio:.2f}x FASTER than CPU\n"
            summary += f"  → NPU achieves {self.npu_metrics.fps:.1f} FPS vs CPU {self.cpu_metrics.fps:.1f} FPS\n"
        else:
            summary += f"  → CPU is {1/self.speedup_ratio:.2f}x FASTER than NPU\n"
            summary += f"  → CPU achieves {self.cpu_metrics.fps:.1f} FPS vs NPU {self.npu_metrics.fps:.1f} FPS\n"
        

        if self.efficiency_gain > 0:
            summary += f"\nEnergy Efficiency Gain: {self.efficiency_gain:.2f}x\n"
            summary += f"Energy Savings: {self.energy_savings_percent:.1f}%\n"
            summary += f"  → NPU: {self.npu_metrics.energy_efficiency_mj_per_frame:.2f} mJ/frame\n"
            summary += f"  → CPU: {self.cpu_metrics.energy_efficiency_mj_per_frame:.2f} mJ/frame\n"
        

        latency_improvement = (self.cpu_metrics.latency_ms - self.npu_metrics.latency_ms) / self.cpu_metrics.latency_ms * 100
        summary += f"\nLatency Improvement: {latency_improvement:.1f}%\n"
        summary += f"  → NPU Latency: {self.npu_metrics.latency_ms:.2f}ms\n"
        summary += f"  → CPU Latency: {self.cpu_metrics.latency_ms:.2f}ms\n"
        

        memory_difference = self.npu_metrics.memory_usage_mb - self.cpu_metrics.memory_usage_mb
        if memory_difference < 0:
            summary += f"\nMemory Savings: {-memory_difference:.1f} MB\n"
        else:
            summary += f"\nMemory Overhead: {memory_difference:.1f} MB\n"
        

        npu_stats = self.npu_metrics.get_statistics()
        cpu_stats = self.cpu_metrics.get_statistics()
        
        if npu_stats and cpu_stats:
            summary += f"\n{'='*80}\n"
            summary += "DETAILED STATISTICS:\n"
            summary += "-" * 40 + "\n"
            summary += f"{'Metric':<20} {'NPU':<15} {'CPU':<15} {'Difference':<15}\n"
            summary += f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}\n"
            
            metrics_data = [
                ("P50 Latency (ms)", npu_stats.get('p50', 0), cpu_stats.get('p50', 0)),
                ("P95 Latency (ms)", npu_stats.get('p95', 0), cpu_stats.get('p95', 0)),
                ("P99 Latency (ms)", npu_stats.get('p99', 0), cpu_stats.get('p99', 0)),
                ("Std Dev (ms)", npu_stats.get('std_dev', 0), cpu_stats.get('std_dev', 0)),
            ]
            
            for name, npu_val, cpu_val in metrics_data:
                diff = npu_val - cpu_val
                diff_str = f"{diff:+.2f}" if diff != 0 else "0.00"
                color = "↓" if diff < 0 else "↑" if diff > 0 else "="
                summary += f"{name:<20} {npu_val:<15.2f} {cpu_val:<15.2f} {color}{diff_str:<14}\n"
        

        summary += f"\n{'='*80}\n"
        summary += "RECOMMENDATION:\n"
        summary += "-" * 40 + "\n"
        summary += self.recommendation + "\n"
        summary += f"{'='*80}\n"
        
        return summary
    
    def to_json(self, filepath: str = None) -> Optional[str]:

        data = {
            'comparison': {
                'timestamp': self.comparison_timestamp,
                'duration_seconds': self.test_duration_seconds,
                'speedup_ratio': self.speedup_ratio,
                'efficiency_gain': self.efficiency_gain,
                'energy_savings_percent': self.energy_savings_percent,
                'recommendation': self.recommendation
            },
            'npu_metrics': self.npu_metrics.to_dict(),
            'cpu_metrics': self.cpu_metrics.to_dict(),
            'statistics': {
                'npu': self.npu_metrics.get_statistics(),
                'cpu': self.cpu_metrics.get_statistics()
            }
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return None
        else:
            return json.dumps(data, indent=2)
    
    def save_report(self, filepath: str = "performance_comparison_report.txt"):

        report = self.get_comparison_summary()
        
        with open(filepath, 'w') as f:
            f.write(report)
        

        json_filepath = filepath.replace('.txt', '.json')
        self.to_json(json_filepath)
        
        return filepath, json_filepath

class AdvantechPerformanceMonitor:

    
    def __init__(self, config: AdvantechConfig, logger: AdvantechLogger):
        self.config = config
        self.logger = logger
        self.metrics = AdvantechPerformanceMetrics()
        self.start_time = None
        self.last_update_time = None
        self.performance_history = deque(maxlen=100) 
        self.lock = threading.Lock()
        
    def start_monitoring(self):

        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time
        self.metrics = AdvantechPerformanceMetrics()
        self.performance_history.clear()
        
    def record_inference(self, inference_time_ms: float, preprocess_time_ms: float = 0,
                        postprocess_time_ms: float = 0):

        with self.lock:

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            

            self.metrics.update(inference_time_ms, preprocess_time_ms, 
                              postprocess_time_ms, memory_mb, cpu_percent)
            

            current_time = time.perf_counter()
            self.performance_history.append({
                'timestamp': current_time,
                'inference_time_ms': inference_time_ms,
                'preprocess_time_ms': preprocess_time_ms,
                'postprocess_time_ms': postprocess_time_ms,
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent
            })
            
            self.last_update_time = current_time
    
    def get_current_metrics(self) -> AdvantechPerformanceMetrics:

        with self.lock:

            current_time = time.perf_counter()
            if self.start_time and len(self.performance_history) > 0:

                recent_records = [r for r in self.performance_history 
                                 if current_time - r['timestamp'] <= 1.0]
                
                if recent_records:
                    avg_inference = sum(r['inference_time_ms'] for r in recent_records) / len(recent_records)
                    avg_preprocess = sum(r['preprocess_time_ms'] for r in recent_records) / len(recent_records)
                    avg_postprocess = sum(r['postprocess_time_ms'] for r in recent_records) / len(recent_records)
                    avg_total = avg_inference + avg_preprocess + avg_postprocess
                    
                    if avg_total > 0:
                        self.metrics.fps = 1000.0 / avg_total
                        self.metrics.throughput_fps = self.metrics.fps
                        self.metrics.latency_ms = avg_total
            
            return self.metrics
    
    def get_performance_history(self, window_seconds: float = 5.0) -> List[Dict]:

        with self.lock:
            current_time = time.perf_counter()
            return [r for r in self.performance_history 
                   if current_time - r['timestamp'] <= window_seconds]
    
    def get_average_performance(self, window_seconds: float = 5.0) -> Dict[str, float]:

        recent = self.get_performance_history(window_seconds)
        
        if not recent:
            return {}
        
        result = {
            'avg_inference_ms': sum(r['inference_time_ms'] for r in recent) / len(recent),
            'avg_preprocess_ms': sum(r['preprocess_time_ms'] for r in recent) / len(recent),
            'avg_postprocess_ms': sum(r['postprocess_time_ms'] for r in recent) / len(recent),
            'avg_memory_mb': sum(r['memory_mb'] for r in recent) / len(recent),
            'avg_cpu_percent': sum(r['cpu_percent'] for r in recent) / len(recent),
            'sample_count': len(recent)
        }
        
        result['avg_total_ms'] = (result['avg_inference_ms'] + 
                                 result['avg_preprocess_ms'] + 
                                 result['avg_postprocess_ms'])
        
        if result['avg_total_ms'] > 0:
            result['fps'] = 1000.0 / result['avg_total_ms']
        
        return result
    
    def stop_monitoring(self) -> AdvantechPerformanceMetrics:

        if self.start_time:
            total_time = time.perf_counter() - self.start_time
            self.metrics.total_time_ms = total_time * 1000
            

            if total_time > 0:
                self.metrics.fps = self.metrics.total_frames / total_time
                self.metrics.throughput_fps = self.metrics.fps
        
        return self.metrics

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
        import glob
        if len(glob.glob('/dev/xclmgmt*')) > 0 or len(glob.glob('/dev/dri/renderD*')) > 0:
            npu_info['available'] = True
        
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
        if npu_info['available']:
            # Try to get CPU info to determine NPU type
            try:
                cpu_info = _get_cpu_info()
                if 'ryzen' in cpu_info.lower() and 'ai' in cpu_info.lower():
                    npu_info['type'] = 'RYZEN_AI'
                elif 'hawk' in cpu_info.lower() and 'point' in cpu_info.lower():
                    npu_info['type'] = 'HAWK_POINT'
                else:
                    npu_info['type'] = 'AMD_NPU'
            except:
                npu_info['type'] = 'AMD_NPU'
            
            # Find xclbin file for supported NPUs
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
                    xclbin_files = [f for f in os.listdir(path) if f.endswith('.xclbin')]
                    if xclbin_files:
                        npu_info['xclbin_path'] = os.path.join(path, xclbin_files[0])
                        npu_info['xclbin_required'] = True
                        break
            
    except Exception as e:
        print(f"NPU hardware detection failed: {e}")
    
    return npu_info

def _get_cpu_info() -> str:
    """Get CPU information"""
    try:
        # Try using lscpu
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Model name' in line:
                    return line.split(':')[1].strip()
        
        # Try reading from /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line.lower():
                    return line.split(':')[1].strip()
    except:
        pass
    return "Unknown CPU"

def get_amd_npu_config(npu_type: str, model_path: str = "") -> Dict[str, Any]:
    """Get configuration based on NPU type"""
    config = {
        'provider': 'VitisAIExecutionProvider',
        'provider_options': {},
        'xclbin_required': False,
        'xclbin_path': ''
    }
    
    # Get model name for precision detection
    model_name = Path(model_path).stem.lower() if model_path else ""
    
    # Set provider options
    provider_options = {
        'cache_dir': str(Path.cwd() / '.vitisai_cache'),
        'cache_key': 'modelcache',
        'enable_cache_file_io_in_mem': '0',
        'target': 'RyzenAI_pso3'  
    }
    

    if npu_type in ['RYZEN_AI', 'HAWK_POINT', 'HPT-XDNA2', 'PHX/HPT', 'PHX', 'HPT']:

        config['xclbin_required'] = False
    else:
        config['xclbin_required'] = True
    
    config['provider_options'] = provider_options
    
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
# Memory Manager
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

# ==========================================================================
# Metrics Collector
# ==========================================================================

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

# ==========================================================================
# Ring Buffer
# ==========================================================================

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
    
    @property
    def dropped_count(self) -> int:
        return self._dropped

# ==========================================================================
# Core Engine Class (Enhanced with Performance Monitoring)
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
        self._model_name = Path(model_path).stem.lower()
        

        self.performance_monitor = AdvantechPerformanceMonitor(config, logger)
        self.performance_monitor.start_monitoring()
        

        self._custom_classes = None
        if config.custom_classes_file and Path(config.custom_classes_file).exists():
            try:
                self._custom_classes = load_custom_classes(config.custom_classes_file, self.task_type.value)
                self.logger.info(f"Loaded custom classes from {config.custom_classes_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load custom classes: {e}")
        
        # Auto-detect task type from model name if not explicitly set
        if self.task_type == AdvantechTaskType.DETECTION:
            if 'seg' in self._model_name:
                self.task_type = AdvantechTaskType.SEGMENTATION
                self.logger.info(f"Auto-detected segmentation model from name: {self._model_name}")
            elif 'cls' in self._model_name:
                self.task_type = AdvantechTaskType.CLASSIFICATION
                self.logger.info(f"Auto-detected classification model from name: {self._model_name}")
    
    @abstractmethod
    def infer(self, frame: np.ndarray) -> Union[List[AdvantechDetection], AdvantechClassification]:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
    
    def warmup(self, iterations: int = 5):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available for warmup")
        
        dummy = np.random.randint(0, 255, (self._model_input_height, self._model_input_width, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)
    
    def get_class_names(self) -> List[str]:

        if self._custom_classes:
            return self._custom_classes
        
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            return IMAGENET_CLASSES
        return COCO_CLASSES
    
    def get_class_name(self, class_id: int) -> str:

        class_names = self.get_class_names()
        
        if class_id < 0:
            return f"invalid_class_{class_id}"
        
        if class_id < len(class_names):
            class_name = class_names[class_id]

            if not class_name or class_name.strip() == "":
                return f"unknown_class_{class_id}"
            return class_name
        else:

            if self.task_type == AdvantechTaskType.CLASSIFICATION:
                if class_id >= 1000 and class_id < 2000:

                    return f"imagenet21k_class_{class_id}"
                else:
                    return f"unknown_class_{class_id}"
            else:
                return f"class_{class_id}"
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:

        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available for preprocessing")
        
        start_time = time.perf_counter()
        
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
        
        preprocess_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
        return np.ascontiguousarray(blob), preprocess_time
    
    def preprocess_classification(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:

        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available for preprocessing")
        
        start_time = time.perf_counter()
        
        target_h, target_w = self._model_input_height, self._model_input_width
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        
        preprocess_time = (time.perf_counter() - start_time) * 1000
        return np.ascontiguousarray(blob), preprocess_time
    
    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int],
                   num_masks: int = 0) -> Tuple[List[AdvantechDetection], float]:

        start_time = time.perf_counter()
        
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        num_cols = output.shape[1]
        num_classes = num_cols - 4 - num_masks
        if num_classes <= 0:
            return [], (time.perf_counter() - start_time) * 1000
        
        boxes = output[:, :4]
        class_scores = output[:, 4:4 + num_classes]
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        valid_mask = confidences >= self.config.confidence_threshold
        if not np.any(valid_mask):
            return [], (time.perf_counter() - start_time) * 1000
        
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
        
        detections = [AdvantechDetection(
            bbox=tuple(boxes_xyxy[i]),
            confidence=float(confidences[i]),
            class_id=int(class_ids[i]),
            class_name=self.get_class_name(int(class_ids[i]))
        ) for i in indices]
        
        postprocess_time = (time.perf_counter() - start_time) * 1000
        return detections, postprocess_time
    
    def postprocess_segmentation(self, det_output: np.ndarray, mask_protos: np.ndarray,
                                 original_shape: Tuple[int, int]) -> Tuple[List[AdvantechDetection], float]:

        start_time = time.perf_counter()
        
        if not CV2_AVAILABLE:
            detections, _ = self.postprocess(det_output, original_shape, num_masks=0)
            return detections, (time.perf_counter() - start_time) * 1000
        
        det_output = det_output.astype(np.float32)
        mask_protos = mask_protos.astype(np.float32)
        
        if det_output.ndim == 3:
            det_output = det_output[0]
        if det_output.shape[0] < det_output.shape[1]:
            det_output = det_output.T
        
        num_cols = det_output.shape[1]
        num_mask_coeffs = 32  # Standard for YOLOv8/11 segmentation
        num_classes = num_cols - 4 - num_mask_coeffs
        
        if num_classes <= 0:
            detections, _ = self.postprocess(det_output, original_shape, num_masks=0)
            return detections, (time.perf_counter() - start_time) * 1000
        
        boxes = det_output[:, :4]
        class_scores = det_output[:, 4:4 + num_classes]
        mask_coeffs = det_output[:, 4 + num_classes:4 + num_classes + num_mask_coeffs]
        
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        valid_mask = confidences >= self.config.confidence_threshold
        
        if not np.any(valid_mask):
            return [], (time.perf_counter() - start_time) * 1000
        
        boxes = boxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]
        mask_coeffs = mask_coeffs[valid_mask]
        
        h_orig, w_orig = original_shape
        scale = self._last_scale
        pad_x, pad_y = self._last_pad_x, self._last_pad_y
        
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = np.clip((boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale, 0, w_orig)
        boxes_xyxy[:, 1] = np.clip((boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale, 0, h_orig)
        boxes_xyxy[:, 2] = np.clip((boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale, 0, w_orig)
        boxes_xyxy[:, 3] = np.clip((boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale, 0, h_orig)
        
        indices = self._nms(boxes_xyxy, confidences, self.config.iou_threshold)[:self.config.max_detections]
        protos = mask_protos[0] if mask_protos.ndim == 4 else mask_protos
        
        detections = []
        
        for idx in indices:
            class_id = int(class_ids[idx])
            x1, y1, x2, y2 = boxes_xyxy[idx]
            
            det = AdvantechDetection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(confidences[idx]),
                class_id=class_id,
                class_name=self.get_class_name(class_id)
            )
            
            # Process mask if available
            if mask_coeffs is not None and idx < len(mask_coeffs):
                try:
                    coeffs = mask_coeffs[idx]
                    # Apply sigmoid to get mask probabilities
                    mask_pred = 1.0 / (1.0 + np.exp(-np.clip(np.tensordot(coeffs, protos, axes=([0], [0])), -50, 50)))
                    
                    # Resize to model input size
                    target_h, target_w = self._model_input_height, self._model_input_width
                    mask_model = cv2.resize(mask_pred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Crop to remove padding
                    scaled_h, scaled_w = int(h_orig * scale), int(w_orig * scale)
                    y_start, y_end = max(0, pad_y), min(target_h, pad_y + scaled_h)
                    x_start, x_end = max(0, pad_x), min(target_w, pad_x + scaled_w)
                    
                    if y_end > y_start and x_end > x_start:
                        mask_cropped = mask_model[y_start:y_end, x_start:x_end]
                        mask_orig = cv2.resize(mask_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                    else:
                        mask_orig = cv2.resize(mask_pred, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                    
                    # Create binary mask
                    mask_binary = (mask_orig > 0.5).astype(np.float32)
                    det.mask = mask_binary
                except Exception as e:
                    self.logger.warning(f"Failed to process mask: {e}")
            
            detections.append(det)
        
        postprocess_time = (time.perf_counter() - start_time) * 1000
        return detections, postprocess_time
    
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

# ==========================================================================
# ONNX Engine (Supports AMD NPU) - Enhanced with Performance Monitoring
# ==========================================================================

class AdvantechOnnxEngine(AdvantechEngine):
    def __init__(self, model_path: str, config: AdvantechConfig, logger: AdvantechLogger, 
                 force_cpu: bool = False):
        super().__init__(model_path, config, logger)
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        

        self.performance_monitor.metrics.model_name = Path(model_path).stem
        
        # Detect AMD NPU
        npu_hardware_info = detect_amd_npu_hardware()
        self.npu_available = npu_hardware_info['available'] and config.use_npu and not force_cpu
        self.npu_type = npu_hardware_info['type']
        
        
        sess_options = ort.SessionOptions()

        sess_options.add_session_config_entry('ep.context_enable', '0')
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 0

        if force_cpu or not config.use_npu:
            self.logger.info(f"CPU-only mode {'forced' if force_cpu else 'enabled (by user request)'}")
            self.npu_available = False
            self.device_type = "CPU"
        else:
            self.device_type = "NPU (if available)"
        

        target_list = [      
            'VAIML',
        ]
        
        user_target = os.environ.get('VITIS_AI_TARGET', '')
        if user_target:
            target_list = [user_target] + target_list
        
        session_created = False
        last_error = None
        

        if self.npu_available and not force_cpu:
            for target_to_try in target_list:
                try:

                    providers = []
                    provider_options = []
                    
                    npu_config = get_amd_npu_config(self.npu_type, model_path)

                    npu_config['provider_options']['target'] = target_to_try
                    

                    if npu_config['xclbin_path'] and os.path.exists(npu_config['xclbin_path']):
                        npu_config['provider_options']['xclbin'] = npu_config['xclbin_path']
                    
                    providers.append('VitisAIExecutionProvider')
                    provider_options.append(npu_config['provider_options'])
                    self.logger.info(f"Trying AMD NPU with target: {target_to_try}")
                    

                    providers.append('CPUExecutionProvider')
                    provider_options.append({})
                    

                    self._session = ort.InferenceSession(
                        model_path, 
                        sess_options=sess_options, 
                        providers=providers,
                        provider_options=provider_options
                    )
                    

                    self._input_name = self._session.get_inputs()[0].name
                    self._output_names = [output.name for output in self._session.get_outputs()]
                    

                    actual_provider = self._session.get_providers()[0]
                    if 'VitisAI' in actual_provider:
                        self.device_type = "AMD NPU"
                        self.logger.info(f"✅ Successfully enabled AMD NPU acceleration (target: {target_to_try})")
                        session_created = True
                        break
                    else:
                        self.logger.info(f"⚠️ Using CPU instead of NPU (target: {target_to_try})")
                        self.device_type = "CPU"
                        session_created = True
                        break
                        
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Target '{target_to_try}' failed: {str(e)[:100]}...")
                    continue
        

        if not session_created:
            if not self.npu_available:
                self.logger.info("NPU not available or CPU-only mode requested, using CPU")
            else:
                self.logger.warning(f"All NPU targets failed, falling back to CPU. Last error: {last_error}")
            
            try:
                self._session = ort.InferenceSession(
                    model_path, 
                    sess_options=sess_options, 
                    providers=['CPUExecutionProvider']
                )

                self._input_name = self._session.get_inputs()[0].name
                self._output_names = [output.name for output in self._session.get_outputs()]
                
                self.device_type = "CPU"
                self.logger.info("✅ CPU mode successful")
                session_created = True
            except Exception as e2:
                raise RuntimeError(f"CPU fallback also failed: {e2}")
        
        if not session_created:
            raise RuntimeError("Failed to create inference session")
        

        self._identify_outputs()
        

        input_shape = self._session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self._model_input_height = input_shape[2]
            self._model_input_width = input_shape[3]
        elif len(input_shape) == 3:
            self._model_input_height = input_shape[1]
            self._model_input_width = input_shape[2]
        

        self.performance_monitor.metrics.device_type = self.device_type
        
        self.logger.info(f"Model input size: {self._model_input_width}x{self._model_input_height}")
        self.logger.info(f"Device type: {self.device_type}")
    
    def _identify_outputs(self):
        """Identify output types for better task handling"""
        outputs = self._session.get_outputs()
        

        if len(outputs) == 1:
            out_shape = outputs[0].shape

            if len(out_shape) == 2 and (out_shape[1] >= 80 or out_shape[1] >= 1000):
                self.task_type = AdvantechTaskType.CLASSIFICATION
                self.logger.info(f"Detected classification model (output shape: {out_shape})")
                return
        

        for idx, out in enumerate(outputs):
            shape = out.shape
            if len(shape) == 4 and (shape[1] == 32 or shape[1] == 160 or 'mask' in out.name.lower()):
                self.task_type = AdvantechTaskType.SEGMENTATION
                self._mask_output_idx = idx
                self.logger.info(f"Detected segmentation model (mask output at index {idx})")
                return
        

        self.task_type = AdvantechTaskType.DETECTION
        self.logger.info("Detected detection model")
    
    def infer(self, frame: np.ndarray) -> Union[List[AdvantechDetection], AdvantechClassification]:
        original_shape = (frame.shape[0], frame.shape[1])
        

        inference_start = time.perf_counter()
        
        # Choose preprocessing based on task type
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            blob, preprocess_time = self.preprocess_classification(frame)
        else:
            blob, preprocess_time = self.preprocess(frame)
        
        # Convert to appropriate precision
        blob = blob.astype(np.float32)
        
        try:

            onnx_start = time.perf_counter()
            

            if hasattr(self, '_input_name'):
                outputs = self._session.run(self._output_names, {self._input_name: blob})
            else:

                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: blob})
            
            onnx_time = (time.perf_counter() - onnx_start) * 1000
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            if self.task_type == AdvantechTaskType.CLASSIFICATION:
                return None
            return []
        
        # Process outputs based on task type
        if self.task_type == AdvantechTaskType.CLASSIFICATION:
            classification_result = self.postprocess_classification(outputs[0])
            postprocess_time = 0 
            result = classification_result
        elif self.task_type == AdvantechTaskType.SEGMENTATION and hasattr(self, '_mask_output_idx'):
            # For segmentation, we typically have two outputs: detection and masks
            if len(outputs) >= 2:
                # Usually detection output is first, masks second
                det_idx = 0
                mask_idx = self._mask_output_idx if hasattr(self, '_mask_output_idx') else 1
                detections, postprocess_time = self.postprocess_segmentation(
                    outputs[det_idx], outputs[mask_idx], original_shape)
                result = detections
            elif len(outputs) == 1:
                # Some segmentation models output combined tensor
                detections, postprocess_time = self.postprocess(outputs[0], original_shape, num_masks=32)
                result = detections
        else:
            # Default to detection
            detections, postprocess_time = self.postprocess(outputs[0], original_shape)
            result = detections
        

        total_inference_time = (time.perf_counter() - inference_start) * 1000
        

        self.performance_monitor.record_inference(
            inference_time_ms=onnx_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time
        )
        
        return result
    
    def postprocess_classification(self, output: np.ndarray) -> AdvantechClassification:

        start_time = time.perf_counter()
        
        if output.ndim > 1:
            output = output.flatten()
        

        probs = np.exp(output - np.max(output))
        probs = probs / probs.sum()
        

        top5_idx = np.argsort(probs)[-5:][::-1]
        

        top_idx = int(top5_idx[0])
        confidence = float(probs[top_idx])
        

        top_name = self.get_class_name(top_idx)
        

        top5_list = []
        for i in top5_idx:
            idx = int(i)
            name = self.get_class_name(idx)
            conf = float(probs[idx])
            top5_list.append((idx, name, conf))
        
        postprocess_time = (time.perf_counter() - start_time) * 1000
        
        return AdvantechClassification(
            class_id=top_idx,
            class_name=top_name,
            confidence=confidence,
            top5=top5_list
        )
    
    def get_performance_metrics(self) -> AdvantechPerformanceMetrics:

        return self.performance_monitor.get_current_metrics()
    
    def cleanup(self):

        final_metrics = self.performance_monitor.stop_monitoring()
        self.logger.info(f"Final performance for {self.device_type}: {final_metrics.fps:.2f} FPS")
        
        self._session = None
        gc.collect()

# ==========================================================================
# PyTorch Engine (CPU only) - Enhanced with Performance Monitoring
# ==========================================================================

class AdvantechPyTorchEngine(AdvantechEngine):
    def __init__(self, model_path: str, config: AdvantechConfig, logger: AdvantechLogger):
        super().__init__(model_path, config, logger)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        self.device_type = "CPU"

        self.performance_monitor.metrics.model_name = Path(model_path).stem
        self.performance_monitor.metrics.device_type = self.device_type
        
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
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available for inference")
        

        inference_start = time.perf_counter()
        

        preprocess_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        

        torch_start = time.perf_counter()
        results = self._model(frame_rgb, conf=self.config.confidence_threshold,
                             iou=self.config.iou_threshold, verbose=False, device=self._device)
        torch_time = (time.perf_counter() - torch_start) * 1000
        
        if not results:
            if self.task_type == AdvantechTaskType.CLASSIFICATION:
                return None
            return []
        
        result = results[0]
        

        postprocess_start = time.perf_counter()
        
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
                

                top_name = self.get_class_name(top_idx)
                
                top5_list = []
                for i in top5_indices:
                    name = self.get_class_name(i)
                    top5_list.append((int(i), name, float(probs_data[i])))
                
                result_obj = AdvantechClassification(
                    class_id=top_idx,
                    class_name=top_name,
                    confidence=top_conf,
                    top5=top5_list
                )
            else:
                result_obj = None
        else:
            boxes = result.boxes
            if boxes is None:
                result_obj = []
            else:
                detections = []
                masks = result.masks.data.cpu().numpy() if result.masks is not None else None
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    det = AdvantechDetection(
                        bbox=tuple(box.xyxy[0].tolist()),
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                        class_name=self.get_class_name(class_id),
                        mask=masks[i] if masks is not None else None
                    )
                    detections.append(det)
                
                result_obj = detections
        
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        

        self.performance_monitor.record_inference(
            inference_time_ms=torch_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time
        )
        
        return result_obj
    
    def get_performance_metrics(self) -> AdvantechPerformanceMetrics:

        return self.performance_monitor.get_current_metrics()
    
    def cleanup(self):

        final_metrics = self.performance_monitor.stop_monitoring()
        self.logger.info(f"Final performance for {self.device_type}: {final_metrics.fps:.2f} FPS")
        
        self._model = None
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

# ==========================================================================
# Model Loader (Enhanced for Performance Comparison)
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
    
    def load(self, model_path: str, model_format: Optional[AdvantechModelFormat] = None, 
             force_cpu: bool = False) -> AdvantechEngine:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if model_format is None:
            model_format = self.detect_format(model_path)
        
        self.logger.info(f"Loading model: {model_path}")
        self.logger.info(f"Model format: {model_format.value}")
        if force_cpu:
            self.logger.info("Force CPU mode enabled")
        
        if model_format == AdvantechModelFormat.ONNX:
            return AdvantechOnnxEngine(model_path, self.config, self.logger, force_cpu)
        elif model_format == AdvantechModelFormat.PYTORCH:
            return AdvantechPyTorchEngine(model_path, self.config, self.logger)
        
        raise ValueError(f"Unsupported format: {model_format}")
    
    def load_for_comparison(self, model_path: str) -> Tuple[AdvantechEngine, AdvantechEngine]:

        self.logger.info("Loading models for performance comparison...")
        

        npu_config = self.config
        npu_config.use_npu = True
        npu_engine = self.load(model_path, force_cpu=False)
        

        cpu_config = self.config
        cpu_config.use_npu = False
        cpu_engine = self.load(model_path, force_cpu=True)
        
        return npu_engine, cpu_engine

# ==========================================================================
# Performance Comparison Runner
# ==========================================================================

class AdvantechPerformanceComparer:

    
    def __init__(self, config: AdvantechConfig, logger: AdvantechLogger):
        self.config = config
        self.logger = logger
        self.model_loader = AdvantechModelLoader(config, logger)
    
    def run_comparison(self, model_path: str, test_frames: int = 100, 
                      warmup_frames: int = 10, video_file: str = None) -> Optional[AdvantechPerformanceComparison]:

        
        print("\n" + "="*80)
        print("ADVANTECH YOLO PERFORMANCE COMPARISON TEST")
        print("="*80)
        print(f"Model: {model_path}")
        
        if video_file:
            print(f"Video File: {video_file}")
            print(f"Processing entire video")
        else:
            print(f"Test Frames: {test_frames}")
            print(f"Warmup Frames: {warmup_frames}")
        
        print("="*80)
        

        npu_info = detect_amd_npu_hardware()
        print(f"\nAMD NPU Detection:")
        print(f"  Available: {npu_info['available']}")
        print(f"  Type: {npu_info['type']}")
        print(f"  Providers: {', '.join(npu_info['providers'])}")
        
        if not npu_info['available']:
            print("\n⚠️ Warning: AMD NPU not detected. Performance comparison may not be meaningful.")
            print("  CPU will be used for both tests.")
        

        if video_file:
            test_frames_list = self._load_video_frames(video_file, test_frames)
            if not test_frames_list:
                print(f"Error: Failed to load video frames from {video_file}")
                return None
            print(f"Loaded {len(test_frames_list)} frames from video")
        else:
            test_frames_list = [self._create_test_image() for _ in range(test_frames)]
        
  
        print("\n" + "-"*80)
        print("Loading NPU Engine...")
        npu_engine = self.model_loader.load(model_path, force_cpu=False)
        

        print(f"Warming up NPU engine ({warmup_frames} frames)...")
        for i in range(min(warmup_frames, len(test_frames_list))):
            npu_engine.infer(test_frames_list[i])
            if (i + 1) % 5 == 0:
                print(f"  NPU warmup: {i + 1}/{min(warmup_frames, len(test_frames_list))}")
        

        print(f"\nTesting NPU performance ({len(test_frames_list)} frames)...")
        npu_start = time.perf_counter()
        
        for i, frame in enumerate(test_frames_list):
            npu_engine.infer(frame)
            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - npu_start
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  NPU: {i + 1}/{len(test_frames_list)} frames, {fps:.1f} FPS")
        
        npu_time = time.perf_counter() - npu_start
        npu_metrics = npu_engine.get_performance_metrics()
        npu_metrics.total_time_ms = npu_time * 1000
        npu_metrics.total_frames = len(test_frames_list)
        
        if npu_time > 0:
            npu_metrics.fps = len(test_frames_list) / npu_time
            npu_metrics.throughput_fps = npu_metrics.fps
        
        npu_engine.cleanup()
        
 
        print("\n" + "-"*80)
        print("Loading CPU Engine...")
        cpu_engine = self.model_loader.load(model_path, force_cpu=True)
        

        print(f"Warming up CPU engine ({warmup_frames} frames)...")
        for i in range(min(warmup_frames, len(test_frames_list))):
            cpu_engine.infer(test_frames_list[i])
            if (i + 1) % 5 == 0:
                print(f"  CPU warmup: {i + 1}/{min(warmup_frames, len(test_frames_list))}")
        

        print(f"\nTesting CPU performance ({len(test_frames_list)} frames)...")
        cpu_start = time.perf_counter()
        
        for i, frame in enumerate(test_frames_list):
            cpu_engine.infer(frame)
            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - cpu_start
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  CPU: {i + 1}/{len(test_frames_list)} frames, {fps:.1f} FPS")
        
        cpu_time = time.perf_counter() - cpu_start
        cpu_metrics = cpu_engine.get_performance_metrics()
        cpu_metrics.total_time_ms = cpu_time * 1000
        cpu_metrics.total_frames = len(test_frames_list)
        
        if cpu_time > 0:
            cpu_metrics.fps = len(test_frames_list) / cpu_time
            cpu_metrics.throughput_fps = cpu_metrics.fps
        
        cpu_engine.cleanup()
        

        comparison = AdvantechPerformanceComparison(
            npu_metrics=npu_metrics,
            cpu_metrics=cpu_metrics,
            test_duration_seconds=max(npu_time, cpu_time)
        )
        

        if self.config.save_performance_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("performance_reports")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f"performance_comparison_{timestamp}.txt"
            json_file = report_dir / f"performance_comparison_{timestamp}.json"
            
            txt_path, json_path = comparison.save_report(str(report_file))
            print(f"\n📊 Performance reports saved:")
            print(f"  Text report: {txt_path}")
            print(f"  JSON report: {json_path}")
        
        return comparison
    
    def _load_video_frames(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:

        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video file {video_path}")
                return []
            
            frame_count = 0
            while True:
                if max_frames and frame_count >= max_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            print(f"Loaded {len(frames)} frames from {video_path}")
            
            if len(frames) == 0:
                print("Warning: No frames loaded from video")
            
        except Exception as e:
            print(f"Error loading video frames: {e}")
            return []
        
        return frames
    
    def _create_test_image(self, width: int = 640, height: int = 480) -> np.ndarray:

        if CV2_AVAILABLE:

            image = np.zeros((height, width, 3), dtype=np.uint8)
            

            for y in range(height):
                for x in range(width):
                    image[y, x] = [
                        int(255 * x / width),      
                        int(255 * y / height),     
                        int(255 * (x + y) / (width + height)) 
                    ]
            

            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            image = cv2.add(image, noise)
            
            return image
        else:

            return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# ==========================================================================
# Renderer (Enhanced for all task types with Performance Display)
# ==========================================================================

class AdvantechOverlayRenderer:
    def __init__(self, config: AdvantechConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {}
        
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID"""
        if class_id not in self.colors:
            self.colors[class_id] = get_color_for_class(class_id)
        return self.colors[class_id]
    
    def render(self, frame: np.ndarray, detections: List[AdvantechDetection],
               metrics: AdvantechMetrics, classification: Optional[AdvantechClassification] = None,
               performance_metrics: Optional[AdvantechPerformanceMetrics] = None) -> np.ndarray:
        if not CV2_AVAILABLE:
            return frame
        
        output = frame.copy()
        

        height, width = output.shape[:2]
        
        # Render segmentation masks (if any)
        
        if self.config.task_type == AdvantechTaskType.SEGMENTATION:
            for det in detections:
                if det.mask is not None and np.any(det.mask > 0):
                    color = self._get_color(det.class_id)
                    mask_resized = cv2.resize(det.mask.astype(np.uint8), 
                                             (output.shape[1], output.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)                    
                   
                    mask_3ch = cv2.merge([mask_resized, mask_resized, mask_resized])
                    mask_3ch = mask_3ch.astype(float) / 255.0                    
              
                    colored_mask = np.zeros_like(output, dtype=float)
                    colored_mask[:] = color
                                                        
                    mask_area = mask_3ch > 0
                    output = output.astype(float)
                    output[mask_area] = output[mask_area] * 0.7 + colored_mask[mask_area] * 0.3
                    output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Render bounding boxes for detection and segmentation
        if self.config.task_type in [AdvantechTaskType.DETECTION, AdvantechTaskType.SEGMENTATION]:
            for det in detections:
                color = self._get_color(det.class_id)
                x1, y1, x2, y2 = map(int, det.bbox)
                
                # Draw bounding box
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f"{det.class_name}: {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
                cv2.rectangle(output, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(output, label, (x1, y1 - 5), self.font, 0.6, (255, 255, 255), 2)
        

        if classification and self.config.task_type == AdvantechTaskType.CLASSIFICATION:

            label = f"{classification.class_name}: {classification.confidence*100:.1f}%"
            cv2.putText(output, "Classification Result:", (10, 50), self.font, 0.8, (0, 255, 255), 2)
            cv2.putText(output, label, (10, 90), self.font, 1.2, (0, 255, 0), 3)
            

            if classification.top5:
                cv2.putText(output, "Top-5 Predictions:", (10, 130), self.font, 0.7, (200, 200, 200), 2)
                y_pos = 170
                for i, (cls_id, cls_name, conf) in enumerate(classification.top5[:5]):
                    text = f"{i+1}. {cls_name}: {conf*100:.1f}%"
                    color = (0, 255, 0) if i == 0 else (255, 255, 255)
                    font_size = 0.6 if i > 0 else 0.7
                    cv2.putText(output, text, (10, y_pos), self.font, font_size, color, 1 if i > 0 else 2)
                    y_pos += 35
        
        # Display FPS and other metrics
        fps_text = f"FPS: {metrics.fps:.1f}"
        cv2.putText(output, fps_text, (width - 150, 30), 
                   self.font, 0.7, (0, 255, 0), 2)
        
        # Display task type in top-left corner
        task_text = f"Task: {self.config.task_type.value}"
        cv2.putText(output, task_text, (10, 30), self.font, 0.7, (255, 255, 255), 2)
        

        if performance_metrics:

            device_text = f"Device: {performance_metrics.device_type}"
            device_color = (0, 255, 0) if "NPU" in performance_metrics.device_type.upper() else (255, 0, 0)
            
            cv2.putText(output, device_text, (10, height - 80), 
                       self.font, 0.6, device_color, 2)
            

            perf_text = f"Latency: {performance_metrics.latency_ms:.1f}ms"
            cv2.putText(output, perf_text, (10, height - 50), 
                       self.font, 0.5, (200, 200, 200), 1)
            

            mem_text = f"Memory: {performance_metrics.memory_usage_mb:.0f}MB"
            cv2.putText(output, mem_text, (10, height - 30), 
                       self.font, 0.5, (200, 200, 200), 1)
        
        return output



__all__ = [
    # Classes
    'AdvantechConfig', 'AdvantechLogger', 'AdvantechTaskType', 'AdvantechModelFormat',
    'AdvantechInputSource', 'AdvantechDetection', 'AdvantechClassification',
    'AdvantechEngine', 'AdvantechOnnxEngine', 'AdvantechPyTorchEngine', 
    'AdvantechModelLoader', 'AdvantechMetricsCollector', 'AdvantechMetrics',
    'AdvantechOverlayRenderer', 'AdvantechMemoryManager', 'AdvantechRingBuffer',
    'AdvantechFrame', 'AdvantechPerformanceMetrics', 'AdvantechPerformanceComparison',
    'AdvantechPerformanceMonitor', 'AdvantechPerformanceComparer',
    
    # Functions
    'detect_amd_npu_hardware', 'get_amd_npu_config',
    
    # Variables
    'ONNX_AVAILABLE', 'TORCH_AVAILABLE', 'CV2_AVAILABLE'
]

if __name__ == "__main__":
    print(f"Advantech YOLO Core Module v{__version__}")
    print(f"Build Date: {__build_date__}")
    print(f"Author: {__author__}")
    print(f"Copyright: {__copyright__}")
    print("\nKey Features:")
    print("  ✅ AMD NPU Acceleration Support")
    print("  ✅ Performance Comparison (NPU vs CPU)")
    print("  ✅ Multi-task Support (Detection, Classification, Segmentation)")
    print("  ✅ Real-time Performance Monitoring")
    print("  ✅ Detailed Performance Reports")

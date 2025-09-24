import base64
import logging
import time
from typing import Optional, Text, Union

import matplotlib.pyplot as plt
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature, notebook

from . import blocks
from .progress import ProgressBar

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml as nvml

    nvml.nvmlInit()
    HAS_NVIDIA_ML_PY = True
except (ImportError, Exception):
    HAS_NVIDIA_ML_PY = False


class SystemMonitor:
    """Monitor system resources including CPU, RAM, GPU, and VRAM."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._check_dependencies()

    def _check_dependencies(self):
        """Log warnings for missing dependencies."""
        if not HAS_PSUTIL:
            self.logger.warning("psutil not available - CPU/RAM monitoring disabled")

    def get_cpu_info(self) -> dict:
        """Get CPU utilization information."""
        if not HAS_PSUTIL:
            return {"cpu_percent": None, "error": "psutil not available"}

        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "cpu_count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg()
                if hasattr(psutil, "getloadavg")
                else None,
            }
        except Exception as e:
            return {"cpu_percent": None, "error": str(e)}

    def get_memory_info(self) -> dict:
        """Get RAM utilization information."""
        if not HAS_PSUTIL:
            return {"memory_percent": None, "error": "psutil not available"}

        try:
            memory = psutil.virtual_memory()
            return {
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
            }
        except Exception as e:
            return {"memory_percent": None, "error": str(e)}

    def get_gpu_info(self) -> dict:
        """Get GPU utilization and VRAM information."""
        gpu_info = {
            "gpu_count": 0,
            "gpus": [],
            "torch_gpu_available": False,
            "torch_gpu_count": 0,
        }

        # Try nvidia-ml-py first (recommended), then pynvml (deprecated)
        if HAS_NVIDIA_ML_PY:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                gpu_info["gpu_count"] = device_count

                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)

                    # Get memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_used_gb = mem_info.used / (1024**3)
                    vram_total_gb = mem_info.total / (1024**3)
                    vram_percent = (mem_info.used / mem_info.total) * 100

                    # Get utilization
                    try:
                        util_rates = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util_rates.gpu
                    except:
                        gpu_util = None

                    # Get name
                    try:
                        name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
                    except:
                        name = f"GPU {i}"

                    gpu_info["gpus"].append(
                        {
                            "id": i,
                            "name": name,
                            "gpu_util_percent": gpu_util,
                            "vram_used_gb": vram_used_gb,
                            "vram_total_gb": vram_total_gb,
                            "vram_percent": vram_percent,
                        }
                    )

                return gpu_info

            except Exception as e:
                gpu_info["nvidia_ml_py_error"] = str(e)

        if not gpu_info["gpus"]:
            gpu_info["error"] = "No GPU monitoring libraries available or no GPUs found"

        return gpu_info

    def get_all_info(self) -> dict:
        """Get all system information."""
        return {
            "timestamp": time.time(),
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "gpu": self.get_gpu_info(),
        }

    def get_system_info(self, prefix: str = "SYSTEM"):
        """Log current system resource usage."""
        info = self.get_all_info()

        # Format CPU info
        cpu_info = info["cpu"]
        if cpu_info.get("cpu_percent") is not None:
            cpu_msg = f"CPU: {cpu_info['cpu_percent']:.1f}%"
            if cpu_info.get("load_avg"):
                cpu_msg += f" (load: {cpu_info['load_avg'][0]:.2f})"
        else:
            cpu_msg = f"CPU: N/A ({cpu_info.get('error', 'unknown error')})"

        # Format memory info
        mem_info = info["memory"]
        if mem_info.get("memory_percent") is not None:
            mem_msg = f"RAM: {mem_info['memory_percent']:.1f}% ({mem_info['memory_used_gb']:.1f}/{mem_info['memory_total_gb']:.1f}GB)"
        else:
            mem_msg = f"RAM: N/A ({mem_info.get('error', 'unknown error')})"

        # Format GPU info
        gpu_info = info["gpu"]
        if gpu_info["gpus"]:
            gpu_msgs = []
            for gpu in gpu_info["gpus"]:
                gpu_util = gpu.get("gpu_util_percent")
                gpu_util_str = f"{gpu_util:.1f}%" if gpu_util is not None else "N/A"
                vram_str = f"{gpu['vram_percent']:.1f}% ({gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f}GB)"
                gpu_msgs.append(f"GPU{gpu['id']}: {gpu_util_str} util, {vram_str} VRAM")
            gpu_msg = " | ".join(gpu_msgs)
        else:
            gpu_msg = f"GPU: N/A ({gpu_info.get('error', 'no GPUs found')})"

        # Log the combined message
        return f"[{prefix}] {cpu_msg} | {mem_msg} | {gpu_msg}"


class Chronometer:
    def __init__(self, unit: Text, progress_bar: Optional[ProgressBar] = None):
        self.unit = unit
        self.progress_bar = progress_bar
        self.current_start_time = None
        self.history = []

    @property
    def is_running(self):
        return self.current_start_time is not None

    def start(self):
        self.current_start_time = time.monotonic()

    def stop(self, do_count: bool = True):
        msg = "No start time available, Did you call stop() before start()?"
        assert self.current_start_time is not None, msg
        end_time = time.monotonic() - self.current_start_time
        self.current_start_time = None
        if do_count:
            self.history.append(end_time)

    def report(self):
        print_fn = print
        if self.progress_bar is not None:
            print_fn = self.progress_bar.write
        print_fn(
            f"Took {np.mean(self.history).item():.3f} "
            f"(+/-{np.std(self.history).item():.3f}) seconds/{self.unit} "
            f"-- ran {len(self.history)} times"
        )


def parse_hf_token_arg(hf_token: Union[bool, Text]) -> Union[bool, Text]:
    if isinstance(hf_token, bool):
        return hf_token
    if hf_token.lower() == "true":
        return True
    if hf_token.lower() == "false":
        return False
    return hf_token


def encode_audio(waveform: np.ndarray) -> Text:
    data = waveform.astype(np.float32).tobytes()
    return base64.b64encode(data).decode("utf-8")


def decode_audio(data: Text) -> np.ndarray:
    # Decode chunk encoded in base64
    byte_samples = base64.decodebytes(data.encode("utf-8"))
    # Recover array from bytes
    samples = np.frombuffer(byte_samples, dtype=np.float32)
    return samples.reshape(1, -1)


def get_padding_left(stream_duration: float, chunk_duration: float) -> float:
    if stream_duration < chunk_duration:
        return chunk_duration - stream_duration
    return 0


def repeat_label(label: Text):
    while True:
        yield label


def get_pipeline_class(class_name: Text) -> type:
    pipeline_class = getattr(blocks, class_name, None)
    msg = f"Pipeline '{class_name}' doesn't exist"
    assert pipeline_class is not None, msg
    return pipeline_class


def get_padding_right(latency: float, step: float) -> float:
    return latency - step


def visualize_feature(duration: Optional[float] = None):
    def apply(feature: SlidingWindowFeature):
        if duration is None:
            notebook.crop = feature.extent
        else:
            notebook.crop = Segment(feature.extent.end - duration, feature.extent.end)
        plt.rcParams["figure.figsize"] = (8, 2)
        notebook.plot_feature(feature)
        plt.tight_layout()
        plt.show()

    return apply


def visualize_annotation(duration: Optional[float] = None):
    def apply(annotation: Annotation):
        extent = annotation.get_timeline().extent()
        if duration is None:
            notebook.crop = extent
        else:
            notebook.crop = Segment(extent.end - duration, extent.end)
        plt.rcParams["figure.figsize"] = (8, 2)
        notebook.plot_annotation(annotation)
        plt.tight_layout()
        plt.show()

    return apply

import torch
import gc
import warnings
from packaging import version


def configure_device():
    """
    Configure PyTorch device (CPU or CUDA) and set precision/memory settings.

    Returns:
        device (torch.device): The selected device.
        precision_dtype (torch.dtype): Selected precision (float32, float16, or bfloat16).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_dtype = torch.float32
    print(f"PyTorch version: {torch.__version__} --- Using device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0).lower()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        major_cc, minor_cc = torch.cuda.get_device_capability(0)

        print(f"[CUDA] GPU: {gpu_name}")
        print(f"[CUDA] CUDA Version: {torch.version.cuda}")
        print(f"[CUDA] GPU Memory: {total_mem:.1f} GB")
        print(f"[CUDA] Compute Capability: {major_cc}.{minor_cc}")

        # Precision control
        if major_cc >= 8:  # Ampere or Ada Lovelace
            precision_dtype = torch.bfloat16
            print("[PRECISION] Using BF16 for Ampere/Ada Lovelace GPU.")
        elif major_cc >= 5:  # Pascal, Turing, etc.
            precision_dtype = torch.float16
            print("[PRECISION] Using FP16 for Pascal or newer GPU.")
        else:
            print("[PRECISION] Using FP32 for older GPU (compute capability < 5.0).")

        # TF32 settings
        if major_cc < 7:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("[TF32] Disabled TF32 for compute capability < 7.0.")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[TF32] Enabled TF32 for compute capability >= 7.0.")

        # cuDNN settings
        torch.backends.cudnn.benchmark = True
        print("[CUDNN] Enabled cuDNN benchmark mode.")

        # Memory recommendations
        if total_mem < 5:
            warnings.warn(
                f"Low GPU memory ({total_mem:.1f} GB) detected. "
                "Recommended batch size: 8–32. Consider reducing model complexity.",
                ResourceWarning
            )
        elif total_mem > 16:
            print(
                f"[INFO] High GPU memory ({total_mem:.1f} GB) detected. "
                "You can use larger batch sizes (e.g., 128–256) or complex models."
            )
        else:
            print(
                f"[INFO] Moderate GPU memory ({total_mem:.1f} GB) detected. "
                "Recommended batch size: 32–64."
            )

        # torch.compile check
        if major_cc >= 8 and version.parse(torch.__version__) >= version.parse("2.0"):
            if hasattr(torch, "compile"):
                print("[OPTIMIZATION] torch.compile is available for model optimization (PyTorch 2.0+).")
            else:
                print("[OPTIMIZATION] torch.compile not available (requires PyTorch 2.0+).")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        if total_mem < 5:
            gc.collect()
            torch.cuda.empty_cache()
            print("[MEMORY] Performed additional memory cleanup for low-end GPU.")

    else:
        print("[CPU] Using CPU for training. Precision set to FP32.")

    return device, precision_dtype


def get_autocast_context(device, precision_dtype):
    """
    Returns appropriate autocast context based on device and selected precision dtype.

    Args:
        device (torch.device): Target device (CPU or CUDA).
        precision_dtype (torch.dtype): Precision type (float32, float16, bfloat16).

    Returns:
        context manager: torch.amp.autocast context.
    """
    enabled = device.type == "cuda" and precision_dtype != torch.float32
    return torch.amp.autocast(
        device_type=device.type,
        dtype=precision_dtype,
        enabled=enabled
    )

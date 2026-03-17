import io
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init


class Tee(io.TextIOBase):
    """A text stream that duplicates writes to multiple underlying streams.

    This is used to mirror stdout/stderr to a log file while keeping the
    original console output unchanged.
    """

    def __init__(self, *streams):
        super().__init__()
        # Filter out None and keep a stable order.
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        # io.TextIOBase requires returning number of characters written.
        # Some streams may return None; we still return len(s).
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        # Preserve tty detection for progress bars etc.
        return any(hasattr(stream, "isatty") and stream.isatty() for stream in self._streams)


class AnsiFilter(io.TextIOBase):
    """Wraps a file stream, stripping ANSI escape codes and resolving
    terminal cursor animations (e.g. progress spinners).

    Handles cursor-up (\\x1b[A) + erase-line (\\x1b[2K) sequences so that
    animated output is collapsed to the final frame only.
    """

    _ANSI_SEQ = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self, stream):
        super().__init__()
        self._stream = stream
        self._pending = []  # completed lines not yet written to file
        self._partial = ""  # current incomplete line (no trailing \n yet)

    def write(self, s):
        n = len(s)
        # Prepend any leftover partial text so split sequences are handled.
        data = self._partial + s
        self._partial = ""

        pos = 0
        for m in self._ANSI_SEQ.finditer(data):
            start, end = m.span()
            self._emit_text(data[pos:start])
            pos = end

            cmd = m.group()[-1]
            param = m.group()[2:-1]

            if cmd == "A":  # Cursor Up
                count = int(param) if param else 1
                for _ in range(count):
                    if self._pending:
                        self._pending.pop()
            elif cmd == "K":  # Erase Line
                self._partial = ""
            # All other sequences (colors, bold, …) are simply dropped.

        self._emit_text(data[pos:])
        self._flush_completed()
        return n

    def _emit_text(self, text):
        """Process plain text (no ANSI sequences), splitting on newlines."""
        if not text:
            return
        parts = text.split("\n")
        self._partial += parts[0]
        for p in parts[1:]:
            self._pending.append(self._resolve_cr(self._partial))
            self._partial = p

    @staticmethod
    def _resolve_cr(line):
        """Handle \\r: keep only the text after the last carriage return."""
        if "\r" in line:
            return line.rsplit("\r", 1)[-1]
        return line

    def _flush_completed(self):
        """Write completed lines, keeping the last one buffered for cursor-up."""
        while len(self._pending) > 1:
            self._stream.write(self._pending.pop(0) + "\n")

    def flush(self):
        for line in self._pending:
            self._stream.write(line + "\n")
        self._pending.clear()
        if self._partial:
            self._stream.write(self._resolve_cr(self._partial))
            self._partial = ""
        self._stream.flush()

    def close(self):
        self.flush()
        self._stream.close()
        super().close()


def setup_console_log(logdir, filename="console.log"):
    """Mirror stdout/stderr to a file under logdir.

    After calling this, anything written to stdout/stderr (print, tracebacks,
    etc.) will be visible both in the terminal and in the log file.

    Returns
    -------
    file handle
        The opened file handle so that the caller can manage its lifetime.
    """
    import sys

    # Line-buffered text file for timely flushing.
    path = logdir / filename
    f = path.open("a", buffering=1)
    filtered = AnsiFilter(f)
    sys.stdout = Tee(sys.stdout, filtered)
    sys.stderr = Tee(sys.stderr, filtered)
    return filtered


def to_np(x):
    return x.detach().cpu().numpy()


def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def weight_init_(m, fan_type="in"):
    # RMSNorm: initialize scale to 1.
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None:
        return

    if weight.numel() == 0:
        return

    # This is a torch private API, but widely used and stable.
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)

    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        # set bias always 0
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


class CudaBenchmark:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class LoggerBackend(ABC):
    """Interface that every logging backend must implement."""

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int):
        pass

    @abstractmethod
    def log_image(self, name: str, value: np.ndarray, step: int):
        pass

    @abstractmethod
    def log_video(self, name: str, value: np.ndarray, step: int):
        pass

    @abstractmethod
    def log_histogram(self, name: str, value: np.ndarray, step: int):
        pass

    @abstractmethod
    def log_text(self, name: str, text: str, step: int):
        pass

    @abstractmethod
    def log_hparams(self, hparams: dict, metrics: dict, run_name: str):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def close(self, exit_code=0):
        pass


class Logger:
    def __init__(self, logdir, filename="metrics.jsonl", backends=None):
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

        if backends is not None:
            self._backends = backends
        else:
            # Default: JSONL + TensorBoard (fully backward compatible)
            self._backends = [
                JSONLBackend(logdir, filename),
                TensorBoardBackend(logdir),
            ]

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def histogram(self, name, value):
        self._histograms[name] = np.array(value)

    def write(self, step, fps=False):
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps/fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        for name, value in scalars:
            for b in self._backends:
                b.log_scalar(name, value, step)
        for name, value in self._images.items():
            for b in self._backends:
                b.log_image(name, value, step)
        for name, value in self._videos.items():
            for b in self._backends:
                b.log_video(name, value, step)
        for name, value in self._histograms.items():
            for b in self._backends:
                b.log_histogram(name, value, step)

        for b in self._backends:
            b.flush()

        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def log_hydra_config(self, config, name="config", step=0, log_hparams=False, hparams_run_name="."):
        """
        Log a Hydra/OmegaConf config to all backends:
          - as YAML text under "{name}/yaml"
          - as flattened hparams
        """
        # 1) Log YAML text
        yaml_str = None
        try:
            from omegaconf import (
                OmegaConf,  # local import to avoid hard dependency at module import
            )

            yaml_str = OmegaConf.to_yaml(config, resolve=True)
        except ImportError:
            # Fallback to string representation
            yaml_str = str(config)
        for b in self._backends:
            b.log_text(f"{name}/yaml", f"```yaml\n{yaml_str}\n```", step)

        # 2) Log flattened hparams
        flat = {}
        container = None
        try:
            from omegaconf import OmegaConf  # local import again

            container = OmegaConf.to_container(config, resolve=True)
        except Exception:
            container = None

        if log_hparams and container is not None:

            def _flatten(prefix, obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatten(f"{prefix}.{k}" if prefix else k, v)
                elif isinstance(obj, (list, tuple)):
                    flat[prefix] = str(obj)
                elif isinstance(obj, (int, float, bool, str)) or obj is None:
                    flat[prefix] = obj if obj is not None else "null"
                else:
                    flat[prefix] = str(obj)

            _flatten("", container)
            for b in self._backends:
                b.log_hparams(flat, {"_": 0}, run_name=hparams_run_name)

    def close(self, exit_code=0):
        for b in self._backends:
            b.close(exit_code=exit_code)


class JSONLBackend(LoggerBackend):
    """Appends scalar metrics as one JSON object per step to a JSONL file.

    Images, videos, histograms, and hparams are silently ignored.
    """

    def __init__(self, logdir, filename="metrics.jsonl"):
        self._path = logdir / filename
        self._pending = {}

    def _entry(self, step):
        if step not in self._pending:
            self._pending[step] = {"step": step}
        return self._pending[step]

    def log_scalar(self, name, value, step):
        self._entry(step)[name] = value

    def log_image(self, name, value, step):
        pass

    def log_video(self, name, value, step):
        pass

    def log_histogram(self, name, value, step):
        pass

    def log_text(self, name, text, step):
        self._entry(step)[name] = text

    def log_hparams(self, hparams, metrics, run_name="."):
        pass

    def flush(self):
        if self._pending:
            with self._path.open("a") as f:
                for entry in self._pending.values():
                    f.write(json.dumps(entry) + "\n")
            self._pending = {}

    def close(self, exit_code=0):
        self.flush()


class TensorBoardBackend(LoggerBackend):
    def __init__(self, logdir):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)

    def log_scalar(self, name, value, step):
        tag = name if "/" in name else "scalars/" + name
        self._writer.add_scalar(tag, value, step)

    def log_image(self, name, value, step):
        self._writer.add_image(name, value, step)

    def log_video(self, name, value, step):
        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)

    def log_histogram(self, name, value, step):
        self._writer.add_histogram(name, value, step)

    def log_text(self, name, text, step):
        self._writer.add_text(name, text, step)

    def log_hparams(self, hparams, metrics, run_name="."):
        import contextlib

        # add_hparams requires a non-empty metrics dict
        with contextlib.suppress(TypeError):
            # Avoid creating a timestamped subdirectory by specifying run_name (PyTorch >= 1.14)
            self._writer.add_hparams(hparams, metrics, run_name=run_name)

    def flush(self):
        self._writer.flush()

    def close(self, exit_code=0):
        self._writer.close()


class WandbBackend(LoggerBackend):
    """W&B backend.  ``import wandb`` only happens when this class is
    instantiated, so it is safe to *define* on machines without wandb.

    Parameters
    ----------
    wandb_cfg : dict
        Dictionary forwarded as keyword arguments to ``wandb.init``.
        Typical keys: ``project``, ``name``, ``config``, ``group``,
        ``tags``, ``notes``, ``mode``, etc.  See the wandb documentation
        for the full list.
    """

    def __init__(self, wandb_cfg: dict):
        import wandb

        self._wandb = wandb
        if wandb.run is None:
            wandb.init(**wandb_cfg)

    def log_scalar(self, name, value, step):
        self._wandb.log({name: value}, step=step)

    def log_image(self, name, value, step):
        self._wandb.log({name: self._wandb.Image(value)}, step=step)

    def log_video(self, name, value, step):
        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        # Tile all batch elements side-by-side; wandb expects (T, C, H, W)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((T, C, H, B * W))
        self._wandb.log(
            {name: self._wandb.Video(value, fps=16, format="gif")},
            step=step,
        )

    def log_histogram(self, name, value, step):
        self._wandb.log({name: self._wandb.Histogram(value)}, step=step)

    def log_text(self, name, text, step):
        self._wandb.log({name: text}, step=step)

    def log_hparams(self, hparams, metrics, run_name="."):
        # wandb config is set at init; update if new keys arrive
        self._wandb.config.update(hparams, allow_val_change=True)

    def flush(self):
        self._wandb.log({}, commit=True)

    def close(self, exit_code=0):
        self._wandb.finish(exit_code=exit_code)


def convert(value, precision=32):
    if isinstance(value, dict):
        return {key: convert(val) for key, val in value.items()}
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


def tensorstats(tensor, prefix):
    return {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
    }


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(obj, path="", optimizers_state_dicts=None, visited=None):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update({k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr})
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(attr, new_path, optimizers_state_dicts, visited)
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


def build_module_tree(module: nn.Module, module_name: str = "") -> dict:
    """Recursively traverse the given nn.Module and build a dictionary with."""
    # 1) Count direct parameters in this module
    direct_param_count = 0
    param_details = {}
    for pname, p in module.named_parameters(recurse=False):
        nump = p.numel()
        param_details[pname] = nump
        direct_param_count += nump

    # 2) Recursively process child modules
    children_info = {}
    for cname, child in module.named_children():
        children_info[cname] = build_module_tree(child, cname)

    # 3) Calculate total parameter count for this module (including all children)
    total = direct_param_count + sum(child["total"] for child in children_info.values())

    return {
        "name": module_name,
        "params": param_details,
        "children": children_info,
        "total": total,
    }


def print_module_tree(info: dict, parent_path: str = "", indent: int = 0):
    """
    Print the module tree built by build_module_tree() in a hierarchical format:
    "(total_parameter_count) (path_to_module_or_param)"
    The function sorts parameters and submodules in descending order of total size.
    """
    # Construct the current path
    name = info["name"]
    if not parent_path:
        full_path = name  # top level
    else:
        if name:  # submodule name is not empty
            full_path = f"{parent_path}/{name}"
        else:
            full_path = parent_path

    # Print total parameter count for the current module
    line = f"{info['total']:11,d} {full_path}"
    print(" " * indent + line)

    # Create a combined list of param_nodes (parameters) and child_nodes (submodules)
    param_nodes = []
    for param_name, count in info["params"].items():
        param_nodes.append({
            "name": param_name,
            "params": {},
            "children": {},
            "total": count,
        })

    child_nodes = list(info["children"].values())

    # Sort by 'total' in descending order
    combined = param_nodes + child_nodes
    combined.sort(key=lambda x: x["total"], reverse=True)

    # Recursively print all children
    for child_info in combined:
        print_module_tree(child_info, full_path, indent + 2)


def compute_rms(tensors):
    """Compute the root mean square (RMS) of a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)


def compute_global_norm(tensors):
    """Compute the global norm (L2 norm) across a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2)


def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x


def print_param_stats(model):
    """
    Prints formatted statistical information of the parameter values (not gradients)
    for the trainable parameters (.requires_grad=True) of the specified PyTorch model.

    - mean
    - std  (population standard deviation: std(unbiased=False))
    - L2 norm (param.data.norm())
    - RMS (root mean square: sqrt(mean(tensor^2)))

    The hierarchical name is displayed by replacing '.' with '/' in the default names
    (e.g., converting "layer.weight" to "layer/weight").
    """

    # List to temporarily store the statistics
    stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            mean_val = data.mean().item()
            std_val = data.std(unbiased=False).item()
            l2_val = data.norm().item()
            rms_val = data.pow(2).mean().sqrt().item()

            hierarchical_name = name.replace(".", "/")
            stats.append((hierarchical_name, mean_val, std_val, l2_val, rms_val))

    # Format function to display numbers in scientific notation with 3 significant digits
    def fmt(v):
        return f"{v:.3e}"

    # Column width settings (adjust if necessary)
    col_widths = [60, 15, 15, 15, 15]
    header_format = (
        f"{{:<{col_widths[0]}}}{{:>{col_widths[1]}}}{{:>{col_widths[2]}}}{{:>{col_widths[3]}}}{{:>{col_widths[4]}}}"
    )
    row_format = header_format

    # Print the header
    print(header_format.format("Parameter", "Mean", "Std", "L2 norm", "RMS"))
    print("-" * (sum(col_widths) + 1))

    # Print the main content
    for hname, mean_val, std_val, l2_val, rms_val in stats:
        print(
            row_format.format(
                hname,
                fmt(mean_val),
                fmt(std_val),
                fmt(l2_val),
                fmt(rms_val),
            )
        )

#!/usr/bin/env python3
# surprise_r2.py
#
# Compute latent surprise on human NetHack data using a trained r2dreamer agent.
#
# Expected usage example:
#
# python surprise_r2.py \
#   --repo_dir /content/r2dreamer \
#   --checkpoint /content/drive/MyDrive/logdir/nethack_clean/latest.pt \
#   --data /content/nle_data/nld-nao-bf/nld-nao-unzipped \
#   --inverse_model /content/drive/MyDrive/nld_aa_inverse_image_only.npz \
#   --output /content/drive/MyDrive/nao_surprise_r2.npz \
#   --override env=nethack \
#   --override env.task=nethack_Score \
#   --override env.env_num=1 \
#   --override env.eval_episode_num=1 \
#   --override model=size12M \
#   --override device=cuda:0 \
#   --seq_length 32 \
#   --horizon 8 \
#   --metric mse
#
# Notes:
# - This script assumes your trained NetHack world model is image-only.
# - It supports either:
#     * a .npy / .npz sequence file
#     * an NLD/NAO dataset directory
# - It supports batch_size=1 for sequence processing.
#
# Output NPZ fields:
#   surprises          object array of per-sequence arrays
#   mean_per_sequence  float32 array
#   metas              object array of metadata dicts
#

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Path/bootstrap helpers
# ------------------------------------------------------------

def add_repo_to_path(repo_dir: str) -> pathlib.Path:
    repo_path = pathlib.Path(repo_dir).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


# ------------------------------------------------------------
# Sequence representation
# ------------------------------------------------------------

@dataclass
class SequenceExample:
    images: np.ndarray            # [T, H, W, C] uint8
    rewards: np.ndarray           # [T] float32
    done: np.ndarray              # [T] bool
    actions: Optional[np.ndarray] = None    # [T-1] int32 if available
    meta: Optional[Dict[str, Any]] = None


class DataLoadingError(RuntimeError):
    pass


def is_numpy_file(path: pathlib.Path) -> bool:
    return path.suffix.lower() in {".npy", ".npz"}


# ------------------------------------------------------------
# TTY rendering
# ------------------------------------------------------------

def tty_to_rgb(chars: np.ndarray, colors: np.ndarray, size=(64, 64)) -> np.ndarray:
    """
    Match the image construction you were already using.
    """
    from PIL import Image

    chars = np.asarray(chars, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)

    rgb = np.zeros((chars.shape[0], chars.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = chars
    rgb[..., 1] = (colors.astype(np.int32) * 16).clip(0, 255).astype(np.uint8)
    rgb[..., 2] = ((chars.astype(np.int32) // 2) + (colors.astype(np.int32) * 8)).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _pick_first(mapping: Dict[str, Any], *names: str):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _extract_images_from_batch(batch: Dict[str, Any], size=(64, 64)) -> np.ndarray:
    """
    Priority:
      1) already rendered image-like tensors
      2) tty_chars + tty_colors -> render
    """
    image = _pick_first(batch, "image", "images", "screen_image", "screen_images", "pixel", "pixels")
    if image is not None:
        image = np.asarray(image)
        if image.ndim == 5:
            if image.shape[0] != 1:
                raise DataLoadingError(f"Expected batch size 1, got image batch shape {image.shape}")
            image = image[0]
        if image.ndim != 4:
            raise DataLoadingError(f"Image tensor must be [T,H,W,C], got {image.shape}")
        return image.astype(np.uint8)

    tty_chars = _pick_first(batch, "tty_chars")
    tty_colors = _pick_first(batch, "tty_colors")
    if tty_chars is not None and tty_colors is not None:
        tty_chars = np.asarray(tty_chars)
        tty_colors = np.asarray(tty_colors)

        if tty_chars.ndim == 4:
            if tty_chars.shape[0] != 1:
                raise DataLoadingError(f"Expected batch size 1 for tty_chars, got {tty_chars.shape}")
            tty_chars = tty_chars[0]
        if tty_colors.ndim == 4:
            if tty_colors.shape[0] != 1:
                raise DataLoadingError(f"Expected batch size 1 for tty_colors, got {tty_colors.shape}")
            tty_colors = tty_colors[0]

        if tty_chars.ndim != 3 or tty_colors.ndim != 3:
            raise DataLoadingError(
                f"tty_chars and tty_colors must be [T,H,W], got {tty_chars.shape} and {tty_colors.shape}"
            )

        images = [tty_to_rgb(tty_chars[t], tty_colors[t], size=size) for t in range(tty_chars.shape[0])]
        return np.stack(images, axis=0).astype(np.uint8)

    raise DataLoadingError("Could not find rendered image or tty observations in NLD batch.")


def _extract_rewards_done_actions(batch: Dict[str, Any], T: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    rewards = _pick_first(batch, "reward", "rewards", "score_delta", "score_deltas", "scores")
    done = _pick_first(batch, "done", "dones", "is_terminal", "terminal", "terminals")
    actions = _pick_first(batch, "action", "actions", "keypresses")

    if rewards is None:
        rewards = np.zeros((T,), dtype=np.float32)
    else:
        rewards = np.asarray(rewards)
        if rewards.ndim == 2:
            if rewards.shape[0] != 1:
                raise DataLoadingError(f"Expected batch size 1 for rewards, got {rewards.shape}")
            rewards = rewards[0]
        rewards = rewards.astype(np.float32)
        if rewards.shape[0] == T:
            rewards = np.diff(rewards, prepend=rewards[:1]).astype(np.float32)

    if done is None:
        done = np.zeros((T,), dtype=bool)
        done[-1] = True
    else:
        done = np.asarray(done)
        if done.ndim == 2:
            if done.shape[0] != 1:
                raise DataLoadingError(f"Expected batch size 1 for done flags, got {done.shape}")
            done = done[0]
        done = done.astype(bool)

    if actions is not None:
        actions = np.asarray(actions)
        if actions.ndim == 2:
            if actions.shape[0] != 1:
                raise DataLoadingError(f"Expected batch size 1 for actions, got {actions.shape}")
            actions = actions[0]
        actions = actions.astype(np.int32)
        if actions.shape[0] == T:
            actions = actions[:-1]
        elif actions.shape[0] != T - 1:
            raise DataLoadingError(
                f"Action sequence must have length T or T-1; got T={T}, action len={actions.shape[0]}"
            )

    if rewards.shape[0] != T:
        raise DataLoadingError(f"Reward length mismatch: expected {T}, got {rewards.shape[0]}")
    if done.shape[0] != T:
        raise DataLoadingError(f"Done length mismatch: expected {T}, got {done.shape[0]}")

    return rewards, done, actions


def load_numpy_sequence(path: pathlib.Path) -> Iterator[SequenceExample]:
    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = set(arr.files)
        required = {"images", "rewards", "done"}
        if not required.issubset(keys):
            raise DataLoadingError(f"NPZ file must contain at least {sorted(required)}, got {sorted(keys)}")
        actions = arr["actions"] if "actions" in keys else None
        yield SequenceExample(
            images=np.asarray(arr["images"], dtype=np.uint8),
            rewards=np.asarray(arr["rewards"], dtype=np.float32),
            done=np.asarray(arr["done"], dtype=bool),
            actions=None if actions is None else np.asarray(actions, dtype=np.int32),
            meta={"source": str(path)},
        )
        return

    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if not isinstance(obj, dict):
            raise DataLoadingError("Object .npy must contain a dict with images/rewards/done.")
        required = {"images", "rewards", "done"}
        if not required.issubset(obj):
            raise DataLoadingError(f"Object .npy must contain at least {sorted(required)}, got {sorted(obj.keys())}")
        yield SequenceExample(
            images=np.asarray(obj["images"], dtype=np.uint8),
            rewards=np.asarray(obj["rewards"], dtype=np.float32),
            done=np.asarray(obj["done"], dtype=bool),
            actions=None if "actions" not in obj else np.asarray(obj["actions"], dtype=np.int32),
            meta={"source": str(path)},
        )
        return

    raise DataLoadingError("Unsupported numpy format. Expected .npz with images/rewards/done or object .npy dict.")


def _register_nld_dataset(nld_module, data_dir: pathlib.Path, dataset_name: str) -> str:
    if not nld_module.db.exists():
        nld_module.db.create()
    try:
        nld_module.add_altorg_directory(str(data_dir), dataset_name)
    except Exception:
        pass
    return dataset_name


def load_nld_sequences(
    data_dir: pathlib.Path,
    dataset_name: Optional[str],
    batch_size: int,
    seq_length: int,
    limit: Optional[int],
    image_size=(64, 64),
) -> Iterator[SequenceExample]:
    try:
        import nle.dataset as nld
    except Exception as exc:
        raise DataLoadingError("Failed to import nle.dataset. Install NLE with dataset support first.") from exc

    ds_name = dataset_name or f"nao-surprise-{data_dir.name}"
    _register_nld_dataset(nld, data_dir, ds_name)

    dataset = None
    constructor_errors = []
    constructor_attempts = [
        dict(batch_size=batch_size, seq_length=seq_length),
        dict(batch_size=batch_size, unroll_length=seq_length),
        dict(batch_size=batch_size),
        {},
    ]
    for kwargs in constructor_attempts:
        try:
            dataset = nld.TtyrecDataset(ds_name, **kwargs)
            break
        except Exception as exc:
            constructor_errors.append((kwargs, repr(exc)))

    if dataset is None:
        msg = "\n".join([f"  kwargs={k}: {e}" for k, e in constructor_errors])
        raise DataLoadingError(f"Could not construct nld.TtyrecDataset for {ds_name}. Attempts:\n{msg}")

    produced = 0
    iterator = iter(dataset)
    while True:
        if limit is not None and produced >= limit:
            return
        try:
            batch = next(iterator)
        except StopIteration:
            return
        except Exception as exc:
            raise DataLoadingError(f"Error while iterating NLD dataset: {exc!r}") from exc

        if not isinstance(batch, dict):
            raise DataLoadingError(f"Expected each NLD batch to be a dict, got {type(batch).__name__}")

        images = _extract_images_from_batch(batch, size=image_size)
        T = images.shape[0]
        rewards, done, actions = _extract_rewards_done_actions(batch, T)

        yield SequenceExample(
            images=images,
            rewards=rewards,
            done=done,
            actions=actions,
            meta={"dataset_name": ds_name, "batch_keys": sorted(batch.keys())},
        )
        produced += 1


# ------------------------------------------------------------
# Inverse model (NumPy MLP inference)
# ------------------------------------------------------------

def load_inverse_model(path: str):
    inv = np.load(path)
    inv_params = {
        "w1": np.asarray(inv["w1"], dtype=np.float32),
        "b1": np.asarray(inv["b1"], dtype=np.float32),
        "w2": np.asarray(inv["w2"], dtype=np.float32),
        "b2": np.asarray(inv["b2"], dtype=np.float32),
        "w3": np.asarray(inv["w3"], dtype=np.float32),
        "b3": np.asarray(inv["b3"], dtype=np.float32),
    }
    input_dim = int(inv["input_dim"])
    num_actions = int(inv["num_actions"])
    return inv_params, input_dim, num_actions


def silu_np(x):
    return x / (1.0 + np.exp(-x))


def mlp_apply_np(params, x):
    x = x @ params["w1"] + params["b1"]
    x = silu_np(x)
    x = x @ params["w2"] + params["b2"]
    x = silu_np(x)
    x = x @ params["w3"] + params["b3"]
    return x


def flatten_obs_part(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
    return x.reshape((x.shape[0], -1))


def pairs_to_input(obs_t, obs_tp1, rew_t, include_reward=True):
    parts = [
        flatten_obs_part(obs_t),
        flatten_obs_part(obs_tp1),
    ]
    if include_reward:
        parts.append(np.asarray(rew_t, np.float32).reshape((-1, 1)))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def predict_inverse_np(inv_params, input_dim, obs_t, obs_tp1, rew_t):
    x = pairs_to_input(obs_t, obs_tp1, rew_t, include_reward=True)
    if x.shape[1] != input_dim:
        raise ValueError(f"Inverse model input dim mismatch: got {x.shape[1]}, expected {input_dim}")
    logits = mlp_apply_np(inv_params, x)
    probs = softmax_np(logits, axis=-1)
    pred = np.argmax(probs, axis=-1).astype(np.int32)
    conf = np.max(probs, axis=-1).astype(np.float32)
    top5 = np.argsort(probs, axis=-1)[:, -5:]
    return pred, conf, top5, probs


# ------------------------------------------------------------
# r2dreamer loading
# ------------------------------------------------------------

def compose_config(repo_dir: pathlib.Path, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    config_dir = repo_dir / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Could not find config dir: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="configs", overrides=overrides)
    return cfg


def load_r2dreamer_agent(repo_dir: pathlib.Path, overrides: list[str], checkpoint_path: str):
    """
    Build Dreamer exactly the same way train.py does:
      - make_envs(config.env) -> obs_space, act_space
      - Dreamer(config.model, obs_space, act_space)
      - load latest.pt["agent_state_dict"]
    """
    cfg = compose_config(repo_dir, overrides)

    from envs import make_envs
    from dreamer import Dreamer

    train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)

    agent = Dreamer(cfg.model, obs_space, act_space).to(cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint is not a dict: {type(ckpt).__name__}")
    if "agent_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'agent_state_dict'. Keys: {list(ckpt.keys())}")

    agent.load_state_dict(ckpt["agent_state_dict"], strict=True)
    agent.eval()

    return agent, cfg, obs_space, act_space, train_envs, eval_envs


# ------------------------------------------------------------
# Obs/action conversion for r2dreamer
# ------------------------------------------------------------

def build_obs_tensor_sequence(
    images: np.ndarray,
    rewards: np.ndarray,
    done: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    images = np.asarray(images, dtype=np.uint8)
    rewards = np.asarray(rewards, dtype=np.float32)
    done = np.asarray(done, dtype=bool)

    if images.ndim != 4:
        raise ValueError(f"images must have shape [T,H,W,C], got {images.shape}")
    T = images.shape[0]

    if rewards.shape != (T,):
        raise ValueError(f"rewards must have shape [{T}], got {rewards.shape}")
    if done.shape != (T,):
        raise ValueError(f"done must have shape [{T}], got {done.shape}")

    is_first = np.zeros((T,), dtype=bool)
    is_first[0] = True
    is_last = done.astype(bool)
    is_terminal = done.astype(bool)

    obs = {
        "image": torch.as_tensor(images[None], dtype=torch.uint8, device=device),
        "reward": torch.as_tensor(rewards[None], dtype=torch.float32, device=device),
        "is_first": torch.as_tensor(is_first[None], dtype=torch.bool, device=device),
        "is_last": torch.as_tensor(is_last[None], dtype=torch.bool, device=device),
        "is_terminal": torch.as_tensor(is_terminal[None], dtype=torch.bool, device=device),
    }
    return obs


def build_prev_action_onehot(action_ids: np.ndarray, act_dim: int, device: torch.device) -> torch.Tensor:
    """
    Convert inferred action ids [T-1] into prev_action tensor [1, T, A].
    Time alignment:
      prev_action[:, 0, :] = 0
      prev_action[:, t, :] = one_hot(action_{t-1}) for t >= 1
    """
    action_ids = np.asarray(action_ids, dtype=np.int64)
    Tm1 = action_ids.shape[0]
    T = Tm1 + 1

    prev = torch.zeros(1, T, act_dim, dtype=torch.float32, device=device)
    if Tm1 > 0:
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=device)
        prev[0, 1:, :] = F.one_hot(ids, num_classes=act_dim).to(torch.float32)
    return prev


# ------------------------------------------------------------
# Surprise computation
# ------------------------------------------------------------

@torch.no_grad()
def compute_surprise_r2(
    agent,
    obs: Dict[str, torch.Tensor],
    prev_action: torch.Tensor,
    horizon: int = 8,
    metric: str = "mse",
) -> np.ndarray:
    """
    Surprise(t) compares:
      imagined latent trajectory from state t using future inferred actions
    against
      posterior latent trajectory from actual observations

    Returns:
      np.ndarray of shape [T-1], or shorter if horizon trims the tail.
    """
    # r2dreamer preprocesses image/reward dtypes/scales internally
    p_obs = agent.preprocess(obs)

    # [B, T, E]
    embed = agent.encoder(p_obs)

    B, T = obs["is_first"].shape
    if B != 1:
        raise ValueError(f"This script currently supports batch size 1 only, got B={B}")

    initial = agent.rssm.initial(B)

    # posterior rollout
    # post_stoch: [B, T, S, K]
    # post_deter: [B, T, D]
    post_stoch, post_deter, _ = agent.rssm.observe(
        embed,
        prev_action,
        initial,
        obs["is_first"],
    )

    # [B, T, F]
    actual_feat = agent.rssm.get_feat(post_stoch, post_deter)

    step_surprise = []

    for t in range(T - 1):
        h = min(horizon, T - 1 - t)

        start_stoch = post_stoch[:, t]      # [B, S, K]
        start_deter = post_deter[:, t]      # [B, D]
        future_actions = prev_action[:, t + 1:t + 1 + h]   # [B, h, A]

        # prior rollout using inferred future actions
        imag_stoch, imag_deter = agent.rssm.imagine_with_action(
            start_stoch,
            start_deter,
            future_actions,
        )

        imag_feat = agent.rssm.get_feat(imag_stoch, imag_deter)           # [B, h, F]
        actual_next = actual_feat[:, t + 1:t + 1 + h]                     # [B, h, F]

        if metric == "mse":
            err = ((imag_feat - actual_next) ** 2).mean(dim=-1)           # [B, h]
        elif metric == "cosine":
            imag_n = F.normalize(imag_feat, dim=-1)
            actual_n = F.normalize(actual_next, dim=-1)
            err = 1.0 - (imag_n * actual_n).sum(dim=-1)                   # [B, h]
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # keep 1-step-ahead error for each start time t
        step_surprise.append(err[:, -1])

    if not step_surprise:
        return np.zeros((0,), dtype=np.float32)

    out = torch.cat(step_surprise, dim=0).detach().cpu().numpy().astype(np.float32)
    return out


def compute_surprise_for_sequence(
    agent,
    ex: SequenceExample,
    inv_params,
    inv_input_dim: int,
    metric: str,
    horizon: int,
    action_mode: str = "inverse",
    random_seed: int = 0,
    seq_idx: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    device = agent.device
    act_dim = agent.act_dim

    obs = build_obs_tensor_sequence(ex.images, ex.rewards, ex.done, device=device)

    rng = np.random.default_rng(random_seed + seq_idx)

    if action_mode == "dataset":
        if ex.actions is None:
            raise ValueError("action_mode=dataset but this sequence has no ground-truth actions.")
        action_ids = np.asarray(ex.actions, dtype=np.int32)
        conf = None
        extra = {
            "used_action_mode": "dataset",
            "inverse_mean_conf": float("nan"),
        }

    elif action_mode == "inverse":
        if ex.actions is None:
            pred_actions, conf, top5, probs = predict_inverse_np(
                inv_params,
                inv_input_dim,
                ex.images[:-1],
                ex.images[1:],
                ex.rewards[:-1],
            )
            action_ids = np.asarray(pred_actions, dtype=np.int32)
            conf = np.asarray(conf, dtype=np.float32)
            extra = {
                "used_action_mode": "inverse",
                "inverse_mean_conf": float(conf.mean()) if len(conf) else float("nan"),
            }
        else:
            action_ids = np.asarray(ex.actions, dtype=np.int32)
            conf = None
            extra = {
                "used_action_mode": "dataset",
                "inverse_mean_conf": float("nan"),
            }

    elif action_mode == "random":
        Tm1 = len(ex.images) - 1
        action_ids = rng.integers(0, agent.act_dim, size=Tm1, dtype=np.int32)
        conf = None
        extra = {
            "used_action_mode": "random",
            "inverse_mean_conf": float("nan"),
        }

    elif action_mode == "zeros":
        Tm1 = len(ex.images) - 1
        action_ids = np.zeros((Tm1,), dtype=np.int32)
        conf = None
        extra = {
            "used_action_mode": "zeros",
            "inverse_mean_conf": float("nan"),
        }

    else:
        raise ValueError(f"Unknown action_mode: {action_mode}")

    prev_action = build_prev_action_onehot(action_ids, act_dim=act_dim, device=device)

    step_surprise = compute_surprise_r2(
        agent=agent,
        obs=obs,
        prev_action=prev_action,
        horizon=horizon,
        metric=metric,
    )

    if conf is not None:
        step_surprise = step_surprise * conf[:len(step_surprise)]

    extra.update({
        "num_steps": int(len(step_surprise)),
        "act_dim": int(act_dim),
    })
    return step_surprise, extra


# ------------------------------------------------------------
# Sequence iterator
# ------------------------------------------------------------

def iter_sequences(args) -> Iterator[SequenceExample]:
    path = pathlib.Path(args.data)
    if path.is_file() and is_numpy_file(path):
        yield from load_numpy_sequence(path)
        return
    if path.is_dir():
        yield from load_nld_sequences(
            data_dir=path,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            limit=args.limit,
            image_size=tuple(args.image_size),
        )
        return
    raise DataLoadingError(f"Unsupported data path: {path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute r2dreamer latent surprise on human demonstrations."
    )
    parser.add_argument("--repo_dir", type=str, required=True,
                        help="Path to the r2dreamer repository root.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to latest.pt or another saved checkpoint.")
    parser.add_argument("--data", type=str, required=True,
                        help="Either a .npy/.npz sequence file or an NLD/NAO dataset directory.")
    parser.add_argument("--inverse_model", type=str, required=True,
                        help="Path to inverse-dynamics .npz file.")
    parser.add_argument("--output", type=str, default="surprises_r2.npz")
    parser.add_argument("--metric", type=str, default="mse", choices=["mse", "cosine"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--action_mode", type=str, default="inverse",
                    choices=["inverse", "random", "zeros", "dataset"])
    parser.add_argument("--random_seed", type=int, default=0)
    # Repeatable hydra overrides
    parser.add_argument("--override", action="append", default=[],
                        help="Hydra override, e.g. --override env=nethack")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size != 1:
        raise ValueError("This first version only supports --batch_size 1.")

    repo_dir = add_repo_to_path(args.repo_dir)

    print("Loading r2dreamer config + checkpoint...")
    agent, cfg, obs_space, act_space, train_envs, eval_envs = load_r2dreamer_agent(
        repo_dir=repo_dir,
        overrides=args.override,
        checkpoint_path=args.checkpoint,
    )

    # Best effort cleanup; we only needed envs to recover spaces consistently.
    try:
        train_envs.close()
    except Exception:
        pass
    try:
        eval_envs.close()
    except Exception:
        pass

    print(f"Loaded agent on device={agent.device}, act_dim={agent.act_dim}")

    inv_params, inv_input_dim, inv_num_actions = load_inverse_model(args.inverse_model)
    print(f"Loaded inverse model: input_dim={inv_input_dim}, num_actions={inv_num_actions}")

    if inv_num_actions != agent.act_dim:
        raise ValueError(
            f"Inverse-model num_actions ({inv_num_actions}) != agent act_dim ({agent.act_dim})."
        )

    all_surprises = []
    metas = []

    for idx, ex in enumerate(iter_sequences(args)):
        seq_surprise, extra = compute_surprise_for_sequence(
            agent=agent,
            ex=ex,
            inv_params=inv_params,
            inv_input_dim=inv_input_dim,
            metric=args.metric,
            horizon=args.horizon,
            action_mode=args.action_mode,
            random_seed=args.random_seed,
            seq_idx=idx,
        )

        all_surprises.append(seq_surprise)

        meta = dict(ex.meta or {})
        meta.update(extra)
        metas.append(meta)

        mean_val = float(seq_surprise.mean()) if len(seq_surprise) else float("nan")
        print(f"[ok] sequence {idx}: len={len(seq_surprise)} mean={mean_val:.6f}")

        if idx % 1000 == 0 and idx > 0:
            np.savez(
                args.output,
                surprises=np.array(all_surprises, dtype=object),
                mean_per_sequence=np.array(
                    [x.mean() if len(x) else np.nan for x in all_surprises],
                    dtype=np.float32,
                ),
                metas=np.array(metas, dtype=object),
            )
            print(f"[checkpoint saved at {idx}]")

    if not all_surprises:
        raise RuntimeError("No valid sequences were processed.")

    out_path = pathlib.Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        surprises=np.array(all_surprises, dtype=object),
        mean_per_sequence=np.array(
            [x.mean() if len(x) else np.nan for x in all_surprises],
            dtype=np.float32,
        ),
        metas=np.array(metas, dtype=object),
    )

    flat_nonempty = [x for x in all_surprises if len(x)]
    if flat_nonempty:
        flat = np.concatenate(flat_nonempty, axis=0)
        print(f"Saved {len(all_surprises)} sequences to {out_path}")
        print(
            f"Global mean={flat.mean():.6f} std={flat.std():.6f} "
            f"min={flat.min():.6f} max={flat.max():.6f}"
        )
    else:
        print(f"Saved {len(all_surprises)} empty sequences to {out_path}")


if __name__ == "__main__":
    main()
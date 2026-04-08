#!/usr/bin/env python3
# rssm_training_human_surprise.py
#
# Fine-tune r2dreamer world model on high-surprise human sequences.
#
# This script:
#   1) Loads a trained Dreamer checkpoint
#   2) Loads surprise results (.npz) from surprise_r2.py
#   3) Selects top surprising sequences
#   4) Re-reads those sequences from NLD/NAO
#   5) Uses the inverse model to infer actions
#   6) Fine-tunes encoder + RSSM (+ optional reward/cont heads)
#
# It does NOT update actor/value by default.
#
# Example:
# python /content/r2dreamer/rssm_training_human_surprise.py \
#   --repo_dir /content/r2dreamer \
#   --checkpoint /content/drive/MyDrive/logdir/nethack_debug/latest.pt \
#   --surprise_npz /content/drive/MyDrive/nao_surprise_r2.npz \
#   --data /content/nle_data/nld-nao-bf/nld-nao-unzipped \
#   --inverse_model /content/drive/MyDrive/nld_aa_inverse_image_only_23.npz \
#   --output_checkpoint /content/drive/MyDrive/logdir/nethack_debug/latest_surprise_ft.pt \
#   --override env=nethack \
#   --override env.task=nethack_Score \
#   --override env.env_num=1 \
#   --override env.eval_episode_num=1 \
#   --override model=size12M \
#   --override device=cuda:0 \
#   --seq_length 32 \
#   --top_k 5000 \
#   --steps 5000 \
#   --batch_size 16 \
#   --lr 1e-4 \
#   --save_every 500
#

import argparse
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Shared helpers
# ============================================================

def add_repo_to_path(repo_dir: str) -> pathlib.Path:
    repo_path = pathlib.Path(repo_dir).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


@dataclass
class SequenceExample:
    images: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    actions: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


class DataLoadingError(RuntimeError):
    pass


def tty_to_rgb(chars: np.ndarray, colors: np.ndarray, size=(64, 64)) -> np.ndarray:
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
    if tty_chars is None or tty_colors is None:
        raise DataLoadingError("Could not find tty observations in NLD batch.")

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
        raise DataLoadingError("Failed to import nle.dataset.") from exc

    ds_name = dataset_name or f"nao-surprise-{data_dir.name}"
    _register_nld_dataset(nld, data_dir, ds_name)

    dataset = None
    constructor_attempts = [
        dict(batch_size=batch_size, seq_length=seq_length),
        dict(batch_size=batch_size, unroll_length=seq_length),
        dict(batch_size=batch_size),
        {},
    ]
    errors = []
    for kwargs in constructor_attempts:
        try:
            dataset = nld.TtyrecDataset(ds_name, **kwargs)
            break
        except Exception as exc:
            errors.append((kwargs, repr(exc)))
    if dataset is None:
        msg = "\n".join([f"  kwargs={k}: {e}" for k, e in errors])
        raise DataLoadingError(f"Could not construct TtyrecDataset. Attempts:\n{msg}")

    produced = 0
    it = iter(dataset)
    while True:
        if limit is not None and produced >= limit:
            return
        try:
            batch = next(it)
        except StopIteration:
            return

        if not isinstance(batch, dict):
            raise DataLoadingError(f"Expected batch dict, got {type(batch).__name__}")

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
    return pred, conf



def compose_config(repo_dir: pathlib.Path, overrides: List[str]):
    from hydra import compose, initialize_config_dir

    config_dir = repo_dir / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Could not find config dir: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="configs", overrides=overrides)
    return cfg


def load_r2dreamer_agent(repo_dir: pathlib.Path, overrides: List[str], checkpoint_path: str):
    cfg = compose_config(repo_dir, overrides)

    from envs import make_envs
    from dreamer import Dreamer

    train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)
    agent = Dreamer(cfg.model, obs_space, act_space).to(cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    if "agent_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'agent_state_dict'. Keys: {list(ckpt.keys())}")
    agent.load_state_dict(ckpt["agent_state_dict"], strict=True)
    agent.eval()

    return agent, cfg, obs_space, act_space, train_envs, eval_envs, ckpt



def build_obs_tensor_sequence(images, rewards, done, device):
    images = np.asarray(images, dtype=np.uint8)
    rewards = np.asarray(rewards, dtype=np.float32)
    done = np.asarray(done, dtype=bool)

    T = images.shape[0]
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
    action_ids = np.asarray(action_ids, dtype=np.int64)
    Tm1 = action_ids.shape[0]
    T = Tm1 + 1
    prev = torch.zeros(1, T, act_dim, dtype=torch.float32, device=device)
    if Tm1 > 0:
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=device)
        prev[0, 1:, :] = F.one_hot(ids, num_classes=act_dim).to(torch.float32)
    return prev



def load_top_sequence_indices(surprise_npz: str, top_k: int) -> np.ndarray:
    data = np.load(surprise_npz, allow_pickle=True)
    means = np.asarray(data["mean_per_sequence"], dtype=np.float32)
    if top_k <= 0 or top_k > len(means):
        top_k = len(means)
    idx = np.argsort(means)[-top_k:]
    idx = np.sort(idx)
    return idx


def materialize_selected_sequences(
    data_dir: pathlib.Path,
    dataset_name: Optional[str],
    seq_length: int,
    selected_indices: np.ndarray,
    image_size=(64, 64),
) -> List[SequenceExample]:
    selected_set = set(int(x) for x in selected_indices.tolist())
    max_needed = int(selected_indices.max()) + 1 if len(selected_indices) else 0

    out = []
    for idx, ex in enumerate(load_nld_sequences(
        data_dir=data_dir,
        dataset_name=dataset_name,
        batch_size=1,
        seq_length=seq_length,
        limit=max_needed,
        image_size=image_size,
    )):
        if idx in selected_set:
            out.append(ex)
        if idx >= max_needed - 1:
            break

    if len(out) != len(selected_indices):
        print(f"[warn] requested {len(selected_indices)} sequences but materialized {len(out)}")
    return out



def make_world_model_optimizer(agent, lr: float):
    params = []
    params += list(agent.encoder.parameters())
    params += list(agent.rssm.parameters())

    if hasattr(agent, "reward"):
        params += list(agent.reward.parameters())
    if hasattr(agent, "cont"):
        params += list(agent.cont.parameters())

    seen = set()
    uniq = []
    for p in params:
        if id(p) not in seen and p.requires_grad:
            uniq.append(p)
            seen.add(id(p))
    return torch.optim.Adam(uniq, lr=lr)


def reward_loss_from_feat(agent, feat, reward_target):
    if not hasattr(agent, "reward"):
        return torch.tensor(0.0, device=feat.device)

    pred = agent.reward(feat)
    # Try common cases: tensor output or distribution-like object with .mean / .mode / .log_prob
    if isinstance(pred, torch.Tensor):
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return F.mse_loss(pred, reward_target)
    if hasattr(pred, "mean"):
        out = pred.mean
        if callable(out):
            out = out()
        if isinstance(out, torch.Tensor):
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            return F.mse_loss(out, reward_target)
    if hasattr(pred, "mode"):
        out = pred.mode
        if callable(out):
            out = out()
        if isinstance(out, torch.Tensor):
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            return F.mse_loss(out, reward_target)

    return torch.tensor(0.0, device=feat.device)


def cont_loss_from_feat(agent, feat, cont_target):
    if not hasattr(agent, "cont"):
        return torch.tensor(0.0, device=feat.device)

    pred = agent.cont(feat)
    if isinstance(pred, torch.Tensor):
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return F.binary_cross_entropy_with_logits(pred, cont_target)

    if hasattr(pred, "log_prob"):
        try:
            return -pred.log_prob(cont_target).mean()
        except Exception:
            pass

    return torch.tensor(0.0, device=feat.device)


def compute_world_model_loss(
    agent,
    obs: Dict[str, torch.Tensor],
    prev_action: torch.Tensor,
    rollout_horizon: int,
    consistency_weight: float,
    reward_weight: float,
    cont_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    p_obs = agent.preprocess(obs)
    embed = agent.encoder(p_obs)

    B, T = obs["is_first"].shape
    initial = agent.rssm.initial(B)

    post_stoch, post_deter, _ = agent.rssm.observe(
        embed,
        prev_action,
        initial,
        obs["is_first"],
    )

    post_feat = agent.rssm.get_feat(post_stoch, post_deter)  # [B,T,F]

    consistency_terms = []
    for t in range(T - 1):
        h = min(rollout_horizon, T - 1 - t)
        start_stoch = post_stoch[:, t]
        start_deter = post_deter[:, t]
        future_actions = prev_action[:, t + 1:t + 1 + h]

        imag_stoch, imag_deter = agent.rssm.imagine_with_action(
            start_stoch,
            start_deter,
            future_actions,
        )
        imag_feat = agent.rssm.get_feat(imag_stoch, imag_deter)
        actual_feat = post_feat[:, t + 1:t + 1 + h]

        consistency_terms.append(((imag_feat - actual_feat) ** 2).mean())

    if consistency_terms:
        consistency_loss = torch.stack(consistency_terms).mean()
    else:
        consistency_loss = torch.tensor(0.0, device=post_feat.device)

    # Predict reward / continuation from posterior features except t=0 alignment
    reward_target = obs["reward"][:, 1:]
    cont_target = (~obs["is_terminal"][:, 1:]).to(torch.float32)

    feat_next = post_feat[:, 1:]  # [B,T-1,F]

    rew_loss = reward_loss_from_feat(agent, feat_next, reward_target)
    con_loss = cont_loss_from_feat(agent, feat_next, cont_target)

    total = (
        consistency_weight * consistency_loss
        + reward_weight * rew_loss
        + cont_weight * con_loss
    )

    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_consistency": float(consistency_loss.detach().cpu()),
        "loss_reward": float(rew_loss.detach().cpu()),
        "loss_cont": float(con_loss.detach().cpu()),
    }
    return total, metrics


def save_checkpoint(
    path: str,
    base_ckpt: Dict[str, Any],
    agent,
    optimizer,
    step: int,
):
    out = dict(base_ckpt)
    out["agent_state_dict"] = agent.state_dict()
    out["surprise_ft_step"] = step
    out["surprise_ft_optimizer_state_dict"] = optimizer.state_dict()
    torch.save(out, path)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune r2dreamer RSSM on human surprise sequences.")
    parser.add_argument("--repo_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--surprise_npz", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--inverse_model", type=str, required=True)
    parser.add_argument("--output_checkpoint", type=str, required=True)

    parser.add_argument("--seq_length", type=int, default=32)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=[64, 64])

    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--rollout_horizon", type=int, default=8)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--reward_weight", type=float, default=1.0)
    parser.add_argument("--cont_weight", type=float, default=1.0)

    parser.add_argument("--override", action="append", default=[])

    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = add_repo_to_path(args.repo_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading Dreamer...")
    agent, cfg, obs_space, act_space, train_envs, eval_envs, base_ckpt = load_r2dreamer_agent(
        repo_dir=repo_dir,
        overrides=args.override,
        checkpoint_path=args.checkpoint,
    )
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
            f"Inverse-model num_actions ({inv_num_actions}) != agent act_dim ({agent.act_dim})"
        )

    print("Selecting high-surprise sequences...")
    selected_indices = load_top_sequence_indices(args.surprise_npz, args.top_k)
    print(f"Selected {len(selected_indices)} top sequences.")

    print("Materializing selected sequences from human dataset...")
    selected_sequences = materialize_selected_sequences(
        data_dir=pathlib.Path(args.data),
        dataset_name=args.dataset_name,
        seq_length=args.seq_length,
        selected_indices=selected_indices,
        image_size=tuple(args.image_size),
    )
    if not selected_sequences:
        raise RuntimeError("No selected sequences could be materialized.")
    print(f"Loaded {len(selected_sequences)} selected sequences.")

    optimizer = make_world_model_optimizer(agent, lr=args.lr)
    agent.train()

    for step in range(1, args.steps + 1):
        batch_examples = random.choices(selected_sequences, k=args.batch_size)

        total_loss = torch.tensor(0.0, device=agent.device)
        metrics_accum = {"loss_total": 0.0, "loss_consistency": 0.0, "loss_reward": 0.0, "loss_cont": 0.0}

        optimizer.zero_grad()

        for ex in batch_examples:
            pred_actions, conf = predict_inverse_np(
                inv_params,
                inv_input_dim,
                ex.images[:-1],
                ex.images[1:],
                ex.rewards[:-1],
            )

            obs = build_obs_tensor_sequence(ex.images, ex.rewards, ex.done, device=agent.device)
            prev_action = build_prev_action_onehot(pred_actions, act_dim=agent.act_dim, device=agent.device)

            loss, metrics = compute_world_model_loss(
                agent=agent,
                obs=obs,
                prev_action=prev_action,
                rollout_horizon=args.rollout_horizon,
                consistency_weight=args.consistency_weight,
                reward_weight=args.reward_weight,
                cont_weight=args.cont_weight,
            )
            total_loss = total_loss + loss / args.batch_size
            for k in metrics_accum:
                metrics_accum[k] += metrics[k] / args.batch_size

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 100.0)
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(
                f"[step {step}] "
                f"total={metrics_accum['loss_total']:.6f} "
                f"cons={metrics_accum['loss_consistency']:.6f} "
                f"rew={metrics_accum['loss_reward']:.6f} "
                f"cont={metrics_accum['loss_cont']:.6f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                path=args.output_checkpoint,
                base_ckpt=base_ckpt,
                agent=agent,
                optimizer=optimizer,
                step=step,
            )
            print(f"[saved] {args.output_checkpoint} at step {step}")

    print("Done.")


if __name__ == "__main__":
    main()
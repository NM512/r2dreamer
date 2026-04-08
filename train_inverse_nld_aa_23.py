import argparse
from dataclasses import dataclass

import numpy as np
import nle.dataset as nld
from PIL import Image


ACTION_SET_23 = [
    13, 107, 108, 106, 104, 117, 110, 98, 121,
    75, 76, 74, 72, 85, 78, 66, 89,
    60, 62, 46, 4, 101, 115,
]
RAW_TO_23 = {a: i for i, a in enumerate(ACTION_SET_23)}


def tty_to_rgb(chars: np.ndarray, colors: np.ndarray, size=(64, 64)) -> np.ndarray:
    chars = np.asarray(chars, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)

    rgb = np.zeros((chars.shape[0], chars.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = chars
    rgb[..., 1] = (colors.astype(np.int32) * 16).clip(0, 255).astype(np.uint8)
    rgb[..., 2] = ((chars.astype(np.int32) // 2) + (colors.astype(np.int32) * 8)).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def silu_np(x):
    return x / (1.0 + np.exp(-x))


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


@dataclass
class InverseMLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    @staticmethod
    def init(input_dim: int, hidden: int, hidden2: int, num_actions: int, rng: np.random.Generator):
        def xavier(in_dim, out_dim):
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            return (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)

        return InverseMLP(
            w1=xavier(input_dim, hidden),
            b1=np.zeros((hidden,), dtype=np.float32),
            w2=xavier(hidden, hidden2),
            b2=np.zeros((hidden2,), dtype=np.float32),
            w3=xavier(hidden2, num_actions),
            b3=np.zeros((num_actions,), dtype=np.float32),
        )

    def forward(self, x: np.ndarray):
        h1 = silu_np(x @ self.w1 + self.b1)
        h2 = silu_np(h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        return h1, h2, logits


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


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def cross_entropy_and_grad(logits: np.ndarray, labels: np.ndarray):
    probs = softmax_np(logits, axis=-1)
    oh = one_hot(labels, logits.shape[1])
    loss = -np.sum(oh * np.log(probs + 1e-8), axis=-1).mean()
    grad_logits = (probs - oh) / logits.shape[0]
    return loss, probs, grad_logits


def collect_filtered_transitions(
    dataset,
    max_batches: int,
    image_size=(64, 64),
    include_reward=True,
):
    x_list = []
    y_list = []
    kept = 0
    skipped = 0

    it = iter(dataset)
    for batch_idx in range(max_batches):
        batch = next(it)

        tty_chars = np.asarray(batch["tty_chars"])
        tty_colors = np.asarray(batch["tty_colors"])
        keypresses = np.asarray(batch["keypresses"])
        scores = np.asarray(batch["scores"])

        if tty_chars.ndim == 4:
            tty_chars = tty_chars[0]
        if tty_colors.ndim == 4:
            tty_colors = tty_colors[0]
        if keypresses.ndim == 2:
            keypresses = keypresses[0]
        if scores.ndim == 2:
            scores = scores[0]

        T = tty_chars.shape[0]
        images = np.stack(
            [tty_to_rgb(tty_chars[t], tty_colors[t], size=image_size) for t in range(T)],
            axis=0,
        ).astype(np.uint8)

        rewards = np.diff(scores.astype(np.float32), prepend=scores[:1].astype(np.float32))
        usable = min(T - 1, len(keypresses))
        for t in range(usable):
            raw_action = int(keypresses[t])
            if raw_action not in RAW_TO_23:
                skipped += 1
                continue

            x = pairs_to_input(
                images[t:t+1],
                images[t+1:t+2],
                rewards[t:t+1],
                include_reward=include_reward,
            )[0]
            y = RAW_TO_23[raw_action]

            x_list.append(x)
            y_list.append(y)
            kept += 1

    if not x_list:
        raise RuntimeError("No filtered transitions were collected.")

    X = np.stack(x_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)

    stats = {
        "kept": kept,
        "skipped": skipped,
        "keep_ratio": kept / max(kept + skipped, 1),
    }
    return X, y, stats


def evaluate(model: InverseMLP, X: np.ndarray, y: np.ndarray, batch_size: int = 1024):
    preds = []
    probs_all = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        _, _, logits = model.forward(xb)
        probs = softmax_np(logits)
        pred = np.argmax(probs, axis=-1)
        preds.append(pred)
        probs_all.append(probs)

    pred = np.concatenate(preds, axis=0)
    probs = np.concatenate(probs_all, axis=0)

    acc = (pred == y).mean()
    top5 = np.argsort(probs, axis=-1)[:, -5:]
    top5_acc = np.mean([y[i] in top5[i] for i in range(len(y))])
    return {
        "acc": float(acc),
        "top5_acc": float(top5_acc),
    }


def train_inverse_on_nld_aa_23(
    dataset,
    steps=3000,
    lr=1e-4,
    hidden=512,
    hidden2=512,
    seed=0,
    include_reward=True,
    log_every=100,
    save_path="nld_aa_inverse_image_only_23.npz",
    max_batches=500,
    image_size=(64, 64),
    batch_size=256,
):
    rng = np.random.default_rng(seed)

    print("Collecting filtered transitions...")
    X, y, stats = collect_filtered_transitions(
        dataset,
        max_batches=max_batches,
        image_size=image_size,
        include_reward=include_reward,
    )
    print(f"Collected X={X.shape}, y={y.shape}")
    print(f"Kept={stats['kept']} Skipped={stats['skipped']} Keep ratio={stats['keep_ratio']:.4f}")

    input_dim = X.shape[1]
    num_actions = len(ACTION_SET_23)

    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    split = int(0.9 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = InverseMLP.init(
        input_dim=input_dim,
        hidden=hidden,
        hidden2=hidden2,
        num_actions=num_actions,
        rng=rng,
    )

    print(f"input_dim={input_dim}, num_actions={num_actions}")
    print(f"train={len(X_train)}, val={len(X_val)}")

    for step in range(1, steps + 1):
        batch_idx = rng.integers(0, len(X_train), size=batch_size)
        xb = X_train[batch_idx]
        yb = y_train[batch_idx]

        h1, h2, logits = model.forward(xb)
        loss, probs, grad_logits = cross_entropy_and_grad(logits, yb)

        grad_w3 = h2.T @ grad_logits
        grad_b3 = grad_logits.sum(axis=0)

        grad_h2 = grad_logits @ model.w3.T

        z1 = xb @ model.w1 + model.b1
        h1 = silu_np(z1)
        z2 = h1 @ model.w2 + model.b2
        h2 = silu_np(z2)

        sig_z2 = 1.0 / (1.0 + np.exp(-z2))
        dsilu_z2 = sig_z2 + z2 * sig_z2 * (1.0 - sig_z2)
        grad_z2 = grad_h2 * dsilu_z2

        grad_w2 = h1.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0)

        grad_h1 = grad_z2 @ model.w2.T
        sig_z1 = 1.0 / (1.0 + np.exp(-z1))
        dsilu_z1 = sig_z1 + z1 * sig_z1 * (1.0 - sig_z1)
        grad_z1 = grad_h1 * dsilu_z1

        grad_w1 = xb.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        model.w1 -= lr * grad_w1.astype(np.float32)
        model.b1 -= lr * grad_b1.astype(np.float32)
        model.w2 -= lr * grad_w2.astype(np.float32)
        model.b2 -= lr * grad_b2.astype(np.float32)
        model.w3 -= lr * grad_w3.astype(np.float32)
        model.b3 -= lr * grad_b3.astype(np.float32)

        if step % log_every == 0 or step == 1:
            train_metrics = evaluate(model, X_train[: min(5000, len(X_train))], y_train[: min(5000, len(y_train))])
            val_metrics = evaluate(model, X_val, y_val) if len(X_val) else {"acc": float("nan"), "top5_acc": float("nan")}
            print(
                f"step={step} loss={loss:.6f} "
                f"train_acc={train_metrics['acc']:.4f} train_top5={train_metrics['top5_acc']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_top5={val_metrics['top5_acc']:.4f}"
            )

    np.savez(
        save_path,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        input_dim=np.int32(input_dim),
        num_actions=np.int32(num_actions),
        action_set=np.asarray(ACTION_SET_23, dtype=np.int32),
        keep_ratio=np.float32(stats["keep_ratio"]),
        kept=np.int32(stats["kept"]),
        skipped=np.int32(stats["skipped"]),
    )
    print(f"Saved inverse model to {save_path}")

    return {
        "model": model,
        "input_dim": input_dim,
        "num_actions": num_actions,
        "stats": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nld-aa-v0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_reward", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="/content/drive/MyDrive/nld_aa_inverse_image_only_23.npz")
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    dataset = nld.TtyrecDataset(args.dataset_name, batch_size=1)
    train_inverse_on_nld_aa_23(
        dataset=dataset,
        steps=args.steps,
        lr=args.lr,
        hidden=args.hidden,
        hidden2=args.hidden2,
        seed=args.seed,
        include_reward=args.include_reward,
        log_every=args.log_every,
        save_path=args.save_path,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )
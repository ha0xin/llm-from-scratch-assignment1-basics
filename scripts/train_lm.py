#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cs336_basics.attention import apply_rope, scaled_dot_product_attention
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.nn import Embedding, Linear, RMSNorm, SwiGLU
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optim import AdamW, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer


def load_vocab(path: str | os.PathLike) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            vocab[int(obj["id"])] = bytes.fromhex(obj["bytes_hex"])
    return vocab


def load_merges(path: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            merges.append((bytes.fromhex(obj["a_hex"]), bytes.fromhex(obj["b_hex"])))
    return merges


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SiLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = h * torch.sigmoid(h)
        return self.w2(h)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        rope_theta: float = 10000.0,
        remove_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_theta = rope_theta
        self.remove_rope = remove_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool, device=device)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if not self.remove_rope:
            token_positions = torch.arange(seq_len, device=x.device).view(1, 1, seq_len)
            q = apply_rope(q, theta=self.rope_theta, token_positions=token_positions)
            k = apply_rope(k, theta=self.rope_theta, token_positions=token_positions)

        mask = self.causal_mask[:seq_len, :seq_len]
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlockExp(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        ffn_type: str,
        remove_rmsnorm: bool,
        use_post_norm: bool,
        remove_rope: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.use_post_norm = use_post_norm
        self.ln1 = IdentityNorm() if remove_rmsnorm else RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln2 = IdentityNorm() if remove_rmsnorm else RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            remove_rope=remove_rope,
            device=device,
            dtype=dtype,
        )
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported ffn_type={ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_post_norm:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
            return x

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLMExp(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        ffn_type: str = "swiglu",
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlockExp(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    context_length=context_length,
                    rope_theta=rope_theta,
                    ffn_type=ffn_type,
                    remove_rmsnorm=remove_rmsnorm,
                    use_post_norm=use_post_norm,
                    remove_rope=remove_rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = IdentityNorm() if remove_rmsnorm else RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.shape[-1] > self.context_length:
            idx = idx[:, -self.context_length :]
        x = self.token_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


@dataclass
class TrainConfig:
    run_name: str
    out_dir: str
    train_data: str
    valid_data: str
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 1024
    rope_theta: float = 10000.0
    ffn_type: str = "swiglu"  # swiglu|silu
    remove_rmsnorm: bool = False
    use_post_norm: bool = False
    remove_rope: bool = False
    batch_size: int = 64
    max_iters: int = 3000
    eval_interval: int = 200
    eval_batches: int = 50
    log_interval: int = 20
    save_interval: int = 500
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    resume: str = ""
    tokenizer_vocab: str = ""
    tokenizer_merges: str = ""
    generate_prompt: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_memmap(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def sample_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    if max_start <= 0:
        raise ValueError(f"Dataset too short: len={len(dataset)} context_length={context_length}")

    starts = np.random.randint(0, max_start, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(context_length + 1, dtype=np.int64)[None, :]
    idx = starts[:, None] + offsets
    chunk = np.asarray(dataset[idx], dtype=np.int64)

    x = torch.from_numpy(chunk[:, :-1]).to(device)
    y = torch.from_numpy(chunk[:, 1:]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: str,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        xb, yb = sample_batch(dataset, batch_size, context_length, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            logits = model(xb)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    context_length: int,
    device: str,
    eos_token: str = "<|endoftext|>",
) -> str:
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    eos_ids = tokenizer.encode(eos_token)
    eos_id = eos_ids[0] if len(eos_ids) == 1 else None

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(logits < v[:, [-1]], torch.tensor(float("-inf"), device=logits.device), logits)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        if eos_id is not None and int(next_id.item()) == eos_id:
            break

    return tokenizer.decode(idx[0].tolist())


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Transformer LM for assignment experiments")
    parser.add_argument("--config", default="", help="Optional config json path")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-dir", default="artifacts/experiments/lm")
    parser.add_argument("--train-data", default="")
    parser.add_argument("--valid-data", default="")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    parser.add_argument("--remove-rmsnorm", action="store_true")
    parser.add_argument("--use-post-norm", action="store_true")
    parser.add_argument("--remove-rope", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--tokenizer-vocab", default="")
    parser.add_argument("--tokenizer-merges", default="")
    parser.add_argument("--generate-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    cfg_dict: dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict.update(json.load(f))

    cli = vars(args)
    for key, value in cli.items():
        if key == "config":
            continue
        if isinstance(value, bool):
            if value:
                cfg_dict[key.replace("-", "_")] = value
        elif value not in ("", None):
            cfg_dict[key.replace("-", "_")] = value

    if not cfg_dict.get("run_name"):
        cfg_dict["run_name"] = time.strftime("run_%Y%m%d_%H%M%S")

    return TrainConfig(**cfg_dict)


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if not cfg.train_data or not cfg.valid_data:
        raise ValueError("Both --train-data and --valid-data must be provided")

    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_tokens = load_memmap(cfg.train_data)
    valid_tokens = load_memmap(cfg.valid_data)

    model = TransformerLMExp(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        ffn_type=cfg.ffn_type,
        remove_rmsnorm=cfg.remove_rmsnorm,
        use_post_norm=cfg.use_post_norm,
        remove_rope=cfg.remove_rope,
        device=torch.device(cfg.device),
        dtype=torch.float32,
    ).to(cfg.device)

    if cfg.compile:
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    start_iter = 0
    best_val = float("inf")
    if cfg.resume:
        start_iter = load_checkpoint(cfg.resume, model=model, optimizer=optimizer)
        best_path = run_dir / "best_metrics.json"
        if best_path.exists():
            with open(best_path, "r", encoding="utf-8") as f:
                best_val = float(json.load(f).get("best_val_loss", best_val))

    autocast_enabled = cfg.device.startswith("cuda") and cfg.dtype == "bfloat16"
    autocast_dtype = torch.bfloat16

    jsonl_path = run_dir / "metrics.jsonl"
    csv_path = run_dir / "metrics.csv"
    csv_headers = [
        "iter",
        "lr",
        "train_loss",
        "val_loss",
        "tokens_seen",
        "elapsed_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    t0 = time.perf_counter()
    tokens_seen = 0

    model.train()
    for it in range(start_iter, cfg.max_iters):
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=cfg.learning_rate,
            min_learning_rate=cfg.min_learning_rate,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.max_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = sample_batch(train_tokens, cfg.batch_size, cfg.context_length, cfg.device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            logits = model(xb)
            loss = cross_entropy(logits.reshape(-1, cfg.vocab_size), yb.reshape(-1))
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        tokens_seen += cfg.batch_size * cfg.context_length

        if it % cfg.log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[iter {it:06d}] loss={loss.item():.4f} lr={lr:.2e} "
                f"tokens_seen={tokens_seen} elapsed={elapsed:.1f}s"
            )

        do_eval = (it % cfg.eval_interval == 0) or (it == cfg.max_iters - 1)
        if do_eval:
            train_loss = estimate_loss(
                model,
                train_tokens,
                batch_size=cfg.batch_size,
                context_length=cfg.context_length,
                eval_batches=max(5, cfg.eval_batches // 5),
                device=cfg.device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            val_loss = estimate_loss(
                model,
                valid_tokens,
                batch_size=cfg.batch_size,
                context_length=cfg.context_length,
                eval_batches=cfg.eval_batches,
                device=cfg.device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            elapsed = time.perf_counter() - t0
            rec = {
                "iter": int(it),
                "lr": float(lr),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "tokens_seen": int(tokens_seen),
                "elapsed_sec": float(elapsed),
            }
            print(
                f"[eval {it:06d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"best_val={best_val:.4f}"
            )

            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(rec)

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, it + 1, run_dir / "checkpoint_best.pt")
                with open(run_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"best_val_loss": best_val, "iter": it}, f, indent=2)

        if (it % cfg.save_interval == 0 and it > start_iter) or (it == cfg.max_iters - 1):
            save_checkpoint(model, optimizer, it + 1, run_dir / "checkpoint_latest.pt")

    summary = {
        "run_name": cfg.run_name,
        "best_val_loss": best_val,
        "max_iters": cfg.max_iters,
        "tokens_seen": tokens_seen,
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Training summary:", json.dumps(summary))

    if cfg.tokenizer_vocab and cfg.tokenizer_merges and cfg.generate_prompt:
        tokenizer = Tokenizer(
            vocab=load_vocab(cfg.tokenizer_vocab),
            merges=load_merges(cfg.tokenizer_merges),
            special_tokens=["<|endoftext|>"],
        )
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=cfg.generate_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            context_length=cfg.context_length,
            device=cfg.device,
        )
        with open(run_dir / "generated.txt", "w", encoding="utf-8") as f:
            f.write(generated + "\n")
        print("Saved generated text to", run_dir / "generated.txt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

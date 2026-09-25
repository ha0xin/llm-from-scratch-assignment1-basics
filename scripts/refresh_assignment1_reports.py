#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LM_DIR = Path("artifacts/experiments/lm")
SLURM_DIR = Path("slurm_logs")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _format_float(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _format_int(x: int | None) -> str:
    if x is None:
        return "-"
    return str(x)


def load_run_info(run_name: str) -> dict[str, Any]:
    run_dir = LM_DIR / run_name
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    metrics = _read_metrics_jsonl(run_dir / "metrics.jsonl")
    val_rows = [m for m in metrics if "val_loss" in m and "iter" in m]

    best_val_loss: float | None = None
    best_iter: int | None = None
    init_val_loss: float | None = None
    final_val_loss: float | None = None
    max_val_loss: float | None = None
    max_val_iter: int | None = None

    if val_rows:
        init_val_loss = float(val_rows[0]["val_loss"])
        final_val_loss = float(val_rows[-1]["val_loss"])
        best_row = min(val_rows, key=lambda r: float(r["val_loss"]))
        worst_row = max(val_rows, key=lambda r: float(r["val_loss"]))
        best_val_loss = float(best_row["val_loss"])
        best_iter = int(best_row["iter"])
        max_val_loss = float(worst_row["val_loss"])
        max_val_iter = int(worst_row["iter"])
    else:
        if "best_val_loss" in summary:
            best_val_loss = float(summary["best_val_loss"])
        if "iter" in summary:
            best_iter = int(summary["iter"])

    if best_val_loss is None and "best_val_loss" in summary:
        best_val_loss = float(summary["best_val_loss"])

    out: dict[str, Any] = {
        "run": run_name,
        "best_val_loss": best_val_loss,
        "best_iter": best_iter,
        "max_iters": int(summary["max_iters"]) if "max_iters" in summary else int(config.get("max_iters", 0) or 0),
        "tokens_seen": int(summary["tokens_seen"]) if "tokens_seen" in summary else None,
        "elapsed_sec": float(summary["elapsed_sec"]) if "elapsed_sec" in summary else None,
        "learning_rate": float(config["learning_rate"]) if "learning_rate" in config else None,
        "batch_size": int(config["batch_size"]) if "batch_size" in config else None,
        "ffn_type": config.get("ffn_type"),
        "d_ff": int(config["d_ff"]) if "d_ff" in config else None,
        "init_val_loss": init_val_loss,
        "final_val_loss": final_val_loss,
        "max_val_loss": max_val_loss,
        "max_val_iter": max_val_iter,
        "has_metrics": bool(metrics),
    }
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_lr_sweep() -> None:
    lr_runs = [
        ("ts_lr_1e4", 1e-4),
        ("ts_lr_3e4", 3e-4),
        ("ts_lr_1e3", 1e-3),
        ("ts_lr_3e3_div", 3e-3),
        ("ts_lr_1e2_div", 1e-2),
        ("ts_lr_3e2_div", 3e-2),
        ("ts_lr_1e1_div", 1e-1),
        ("ts_lr_3e1_div_probe2", 3e-1),
        ("ts_lr_1e0_div_probe2", 1.0),
        ("ts_lr_3e0_div_probe3", 3.0),
    ]

    rows: list[dict[str, Any]] = []
    for run, lr in lr_runs:
        rec = load_run_info(run)
        rec["lr"] = lr
        rows.append(rec)

    rows.sort(key=lambda x: float(x["lr"]))
    out_json = []
    for r in rows:
        out_json.append(
            {
                "run": r["run"],
                "lr": r["lr"],
                "best_val_loss": r["best_val_loss"],
                "best_iter": r["best_iter"],
                "max_iters": r["max_iters"],
                "tokens_seen": r["tokens_seen"],
                "elapsed_sec": r["elapsed_sec"],
            }
        )
    write_json(LM_DIR / "lr_sweep_summary.json", out_json)

    md_lines = [
        "# Learning Rate Sweep Summary",
        "",
        "| run | lr | best_val_loss | best_iter | max_iters | tokens_seen | elapsed_sec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + f"{r['run']} | {r['lr']:.1e} | {_format_float(r['best_val_loss'])} | {_format_int(r['best_iter'])} | "
            + f"{_format_int(r['max_iters'])} | {_format_int(r['tokens_seen'])} | {_format_float(r['elapsed_sec'], 2)} |"
        )
    write_markdown(LM_DIR / "lr_sweep_summary.md", md_lines)

    div_rows: list[dict[str, Any]] = []
    for r in rows:
        init_v = r["init_val_loss"]
        final_v = r["final_val_loss"]
        max_v = r["max_val_loss"]
        if init_v is None or final_v is None or max_v is None:
            continue
        ratio_final = float(final_v) / float(init_v) if init_v else None
        ratio_max = float(max_v) / float(init_v) if init_v else None
        status = "stable_or_improved"
        if ratio_final is not None and ratio_final >= 2.0:
            status = "divergent"
        elif ratio_max is not None and ratio_max >= 10.0:
            status = "divergent_spike"
        elif ratio_max is not None and ratio_max >= 3.0:
            status = "unstable_spike"
        div_rows.append(
            {
                "run": r["run"],
                "lr": r["lr"],
                "init_val_loss": init_v,
                "final_val_loss": final_v,
                "max_val_loss": max_v,
                "max_val_iter": r["max_val_iter"],
                "ratio_final_over_init": ratio_final,
                "ratio_max_over_init": ratio_max,
                "status": status,
            }
        )

    write_json(LM_DIR / "lr_divergence_summary.json", div_rows)
    md_lines = [
        "# Learning Rate Divergence Summary",
        "",
        "| run | lr | init_val | final_val | max_val | max_iter | final/init | max/init | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in div_rows:
        md_lines.append(
            "| "
            + f"{r['run']} | {r['lr']:.1e} | {_format_float(r['init_val_loss'])} | {_format_float(r['final_val_loss'])} | "
            + f"{_format_float(r['max_val_loss'])} | {_format_int(r['max_val_iter'])} | "
            + f"{_format_float(r['ratio_final_over_init'], 2)} | {_format_float(r['ratio_max_over_init'], 2)} | {r['status']} |"
        )
    write_markdown(LM_DIR / "lr_divergence_summary.md", md_lines)


def build_batch_summary() -> None:
    runs = [
        ("bs1_lr3e4_5k", "ts_bs1_lr3e4_5k", 1),
        ("bs32_lr1e3_5k", "ts_bs32_lr1e3_5k", 32),
        ("bs64_lr1e3_5k", "ts_lr_1e3", 64),
        ("bs128_lr15e3_5k", "ts_bs128_lr15e3_5k", 128),
        ("bs256_limit_probe", "ts_bs256_limit_probe", 256),
    ]
    rows: list[dict[str, Any]] = []
    for tag, run, bs in runs:
        rec = load_run_info(run)
        rec["tag"] = tag
        rec["batch_size"] = bs
        rec["status"] = "ok" if rec["best_val_loss"] is not None else "missing"
        rec["note"] = ""
        rows.append(rec)

    err_path = SLURM_DIR / "train_lm_1475.err"
    oom_note = ""
    if err_path.exists():
        content = err_path.read_text(encoding="utf-8", errors="ignore")
        if "OutOfMemoryError" in content:
            # Keep the note short for table readability.
            oom_note = "CUDA OOM at first forward pass (batch=512)"
        else:
            oom_note = "failed; see slurm_logs/train_lm_1475.err"
    else:
        oom_note = "log missing"

    rows.append(
        {
            "tag": "bs512_limit_probe",
            "run": "ts_bs512_limit_probe",
            "batch_size": 512,
            "status": "oom",
            "best_val_loss": None,
            "best_iter": None,
            "max_iters": 300,
            "tokens_seen": None,
            "elapsed_sec": None,
            "note": oom_note,
        }
    )
    rows.sort(key=lambda x: int(x["batch_size"]))

    out_json = []
    for r in rows:
        out_json.append(
            {
                "tag": r["tag"],
                "run": r["run"],
                "batch_size": r["batch_size"],
                "status": r["status"],
                "best_val_loss": r["best_val_loss"],
                "best_iter": r["best_iter"],
                "max_iters": r["max_iters"],
                "tokens_seen": r["tokens_seen"],
                "elapsed_sec": r["elapsed_sec"],
                "note": r["note"],
            }
        )
    write_json(LM_DIR / "batch_size_summary.json", out_json)

    md_lines = [
        "# Batch Size Summary",
        "",
        "| batch_size | tag | run | status | best_val_loss | tokens_seen | note |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + f"{r['batch_size']} | {r['tag']} | {r['run']} | {r['status']} | "
            + f"{_format_float(r.get('best_val_loss'))} | {_format_int(r.get('tokens_seen'))} | {r.get('note', '')} |"
        )
    write_markdown(LM_DIR / "batch_size_summary.md", md_lines)


def build_ablation_summary() -> None:
    runs = [
        ("baseline_lr1e3_5k", "ts_lr_1e3"),
        ("no_rmsnorm_lr1e3_5k", "ts_ablate_no_rmsnorm_lr1e3_5k"),
        ("no_rmsnorm_lr3e4_5k", "ts_ablate_no_rmsnorm_lr3e4_5k"),
        ("postnorm_lr1e3_5k", "ts_ablate_postnorm_lr1e3_5k"),
        ("no_rope_lr1e3_5k", "ts_ablate_no_rope_lr1e3_5k"),
        ("silu_lr1e3_5k_dff2048", "ts_ablate_silu_dff2048_lr1e3_5k"),
    ]
    rows: list[dict[str, Any]] = []
    for tag, run in runs:
        rec = load_run_info(run)
        rec["tag"] = tag
        rows.append(rec)

    out_json = []
    for r in rows:
        out_json.append(
            {
                "tag": r["tag"],
                "run": r["run"],
                "best_val_loss": r["best_val_loss"],
                "best_iter": r["best_iter"],
                "max_iters": r["max_iters"],
                "tokens_seen": r["tokens_seen"],
                "elapsed_sec": r["elapsed_sec"],
                "ffn_type": r["ffn_type"],
                "d_ff": r["d_ff"],
            }
        )
    write_json(LM_DIR / "ablation_summary.json", out_json)

    md_lines = [
        "# Ablation Summary",
        "",
        "| tag | run | best_val_loss | ffn_type | d_ff | tokens_seen |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + f"{r['tag']} | {r['run']} | {_format_float(r['best_val_loss'])} | "
            + f"{r.get('ffn_type', '-')} | {_format_int(r.get('d_ff'))} | {_format_int(r.get('tokens_seen'))} |"
        )
    write_markdown(LM_DIR / "ablation_summary.md", md_lines)


def main() -> int:
    build_lr_sweep()
    build_batch_summary()
    build_ablation_summary()
    print("Refreshed assignment1 summary artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

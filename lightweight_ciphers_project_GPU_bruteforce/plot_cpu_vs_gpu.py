#!/usr/bin/env python3
"""
plot_cpu_vs_gpu.py  —  Compare CPU-optimised vs GPU brute-force results.

Loads two CSVs:
  --cpu_csv   results_CPU_optimizations.csv   (cpu_optimized_* variants)
  --gpu_csv   results_all.csv                 (cpu_baseline + gpu* variants)

Generates per-cipher plots AND cross-cipher summary plots in --outdir:

  Per-cipher:
    <cipher>__cpu_vs_gpu_throughput.png    — keys/s vs bits: baseline / cpu-opt / best-GPU
    <cipher>__cpu_vs_gpu_time.png          — seconds vs bits (same three lines)

  Summary:
    summary__cpu_opt_speedup.png           — CPU-opt speedup over CPU baseline (bar)
    summary__gpu_speedup_over_cpu_opt.png  — Best-GPU speedup over CPU-opt (bar)
    summary__throughput_comparison.png     — Grouped bar: baseline / cpu-opt / best-GPU at max bits
    summary__all_throughput_overlay.png    — All ciphers: cpu-opt vs best-GPU throughput lines

Usage:
  python3 plot_cpu_vs_gpu.py --cpu_csv ../results_CPU_optimizations.csv \\
                              --gpu_csv ../results_all.csv \\
                              --outdir  plots_cpu_vs_gpu
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── styling ──────────────────────────────────────────────────────────────────

CIPHER_DISPLAY = {
    "simon32_64":     "SIMON 32/64",
    "present80":      "PRESENT-80",
    "speck64_128":    "SPECK 64/128",
    "grain_v1":       "Grain v1",
    "trivium":        "Trivium",
    "chacha20":       "ChaCha20",
    "tinyjambu_128":  "TinyJAMBU-128",
    "zuc_128":        "ZUC-128",
    "snow_v":         "SNOW-V",
    "aes128":         "AES-128",
    "salsa20":        "Salsa20",
    "grain128aeadv2": "Grain-128AEADv2",
    "rocca":          "Rocca",
    "rocca_s":        "Rocca-S",
}

CIPHER_COLOURS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896",
]

LINE_STYLES = {
    "cpu_baseline": dict(linestyle=":", marker="^", linewidth=1.3, color="#7f7f7f"),
    "cpu_opt":      dict(linestyle="--", marker="s", linewidth=1.6, color="#1f77b4"),
    "best_gpu":     dict(linestyle="-",  marker="o", linewidth=2.0, color="#d62728"),
}

def cd(name):
    return CIPHER_DISPLAY.get(name, name)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def safe_label(s):
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)

# ── data loading ──────────────────────────────────────────────────────────────

CPU_OPT_PREFIXES = ("cpu_optimized",)

def is_cpu_opt(variant):
    return any(variant.startswith(p) for p in CPU_OPT_PREFIXES)

def load_csvs(cpu_csv_path, gpu_csv_path):
    """Return a merged DataFrame with a 'role' column: baseline / cpu_opt / gpu_*."""
    needed = ["cipher","platform","variant","unknown_bits","seconds","keys_per_sec"]

    cpu_df = pd.read_csv(cpu_csv_path)
    gpu_df = pd.read_csv(gpu_csv_path)

    for df, name in [(cpu_df, cpu_csv_path), (gpu_df, gpu_csv_path)]:
        miss = [c for c in needed if c not in df.columns]
        if miss:
            raise SystemExit(f"{name} missing columns: {miss}")

    # Assign roles
    cpu_df["role"] = cpu_df["variant"].apply(
        lambda v: "cpu_opt" if is_cpu_opt(v) else "cpu_baseline"
    )
    gpu_df["role"] = gpu_df.apply(
        lambda r: "cpu_baseline" if r["platform"] == "cpu" else "gpu",
        axis=1
    )

    df = pd.concat([cpu_df, gpu_df], ignore_index=True)

    for col in ["unknown_bits", "seconds", "keys_per_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["unknown_bits","seconds","keys_per_sec"])
    df = df[df["unknown_bits"] > 0]
    df = df[df["cipher"] != "cipher"]   # drop any duplicate header rows
    return df


def best_gpu(df, cipher):
    """Return the single best-GPU series (variant with highest median keys_per_sec)."""
    g = df[(df["cipher"] == cipher) & (df["platform"] == "gpu")]
    if g.empty:
        return pd.DataFrame()
    best_var = g.groupby("variant")["keys_per_sec"].median().idxmax()
    return g[g["variant"] == best_var].sort_values("unknown_bits")


def cpu_opt_series(df, cipher):
    """Return the CPU-opt series (highest median among cpu_opt variants)."""
    g = df[(df["cipher"] == cipher) & (df["role"] == "cpu_opt")]
    if g.empty:
        return pd.DataFrame()
    best_var = g.groupby("variant")["keys_per_sec"].median().idxmax()
    return g[g["variant"] == best_var].sort_values("unknown_bits")


def cpu_baseline_series(df, cipher):
    """Return the CPU baseline series."""
    g = df[(df["cipher"] == cipher) & (df["role"] == "cpu_baseline")
           & (df["platform"] == "cpu")]
    if g.empty:
        return pd.DataFrame()
    best_var = g.groupby("variant")["keys_per_sec"].median().idxmax()
    return g[g["variant"] == best_var].sort_values("unknown_bits")


# ── per-cipher plots ──────────────────────────────────────────────────────────

def per_cipher_throughput(df, cipher, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))

    base = cpu_baseline_series(df, cipher)
    opt  = cpu_opt_series(df, cipher)
    gpu  = best_gpu(df, cipher)

    plotted = False
    if not base.empty:
        ax.plot(base["unknown_bits"], base["keys_per_sec"],
                label="CPU Baseline", **LINE_STYLES["cpu_baseline"])
        plotted = True
    if not opt.empty:
        lbl = "CPU Optimised (" + opt["variant"].iloc[0].replace("cpu_optimized_","") + ")"
        ax.plot(opt["unknown_bits"], opt["keys_per_sec"],
                label=lbl, **LINE_STYLES["cpu_opt"])
        plotted = True
    if not gpu.empty:
        lbl = "Best GPU (" + gpu["variant"].iloc[0] + ")"
        ax.plot(gpu["unknown_bits"], gpu["keys_per_sec"],
                label=lbl, **LINE_STYLES["best_gpu"])
        plotted = True

    if not plotted:
        plt.close(fig); return

    ax.set_yscale("log")
    ax.set_title(f"{cd(cipher)} — Throughput: CPU Opt vs GPU", fontsize=13)
    ax.set_xlabel("Unknown key bits", fontsize=11)
    ax.set_ylabel("Throughput (keys/s, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, f"{safe_label(cipher)}__cpu_vs_gpu_throughput.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  saved: {path}")


def per_cipher_time(df, cipher, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))

    base = cpu_baseline_series(df, cipher)
    opt  = cpu_opt_series(df, cipher)
    gpu  = best_gpu(df, cipher)

    plotted = False
    if not base.empty:
        ax.plot(base["unknown_bits"], base["seconds"],
                label="CPU Baseline", **LINE_STYLES["cpu_baseline"])
        plotted = True
    if not opt.empty:
        lbl = "CPU Optimised (" + opt["variant"].iloc[0].replace("cpu_optimized_","") + ")"
        ax.plot(opt["unknown_bits"], opt["seconds"],
                label=lbl, **LINE_STYLES["cpu_opt"])
        plotted = True
    if not gpu.empty:
        lbl = "Best GPU (" + gpu["variant"].iloc[0] + ")"
        ax.plot(gpu["unknown_bits"], gpu["seconds"],
                label=lbl, **LINE_STYLES["best_gpu"])
        plotted = True

    if not plotted:
        plt.close(fig); return

    ax.set_yscale("log")
    ax.set_title(f"{cd(cipher)} — Attack Time: CPU Opt vs GPU", fontsize=13)
    ax.set_xlabel("Unknown key bits", fontsize=11)
    ax.set_ylabel("Time (seconds, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, f"{safe_label(cipher)}__cpu_vs_gpu_time.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  saved: {path}")


# ── summary: grouped throughput bar chart ─────────────────────────────────────

def summary_throughput_comparison(df, ciphers, outdir):
    """
    Grouped bar chart: CPU-baseline / CPU-opt / Best-GPU throughput at maximum bits
    measured per cipher (uses the max bits common to all three, or max available).
    """
    names, base_vals, opt_vals, gpu_vals = [], [], [], []

    for cipher in ciphers:
        base = cpu_baseline_series(df, cipher)
        opt  = cpu_opt_series(df, cipher)
        gpu  = best_gpu(df, cipher)
        if base.empty and opt.empty and gpu.empty:
            continue

        def max_kps(s):
            return float(s["keys_per_sec"].iloc[s["unknown_bits"].argmax()]) if not s.empty else 0.0

        names.append(cd(cipher))
        base_vals.append(max_kps(base))
        opt_vals.append(max_kps(opt))
        gpu_vals.append(max_kps(gpu))

    if not names:
        return

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w, base_vals, w, label="CPU Baseline",  color="#7f7f7f", edgecolor="black", lw=0.5)
    b2 = ax.bar(x,     opt_vals,  w, label="CPU Optimised", color="#1f77b4", edgecolor="black", lw=0.5)
    b3 = ax.bar(x + w, gpu_vals,  w, label="Best GPU",      color="#d62728", edgecolor="black", lw=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (keys/s, log scale)", fontsize=11)
    ax.set_title("Throughput at Maximum Measured Bits — CPU Baseline vs CPU Opt vs Best GPU",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    path = os.path.join(outdir, "summary__throughput_comparison.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  saved: {path}")


# ── summary: CPU-opt speedup over CPU baseline ────────────────────────────────

def summary_cpu_opt_speedup(df, ciphers, outdir):
    records = []
    for cipher in ciphers:
        base = cpu_baseline_series(df, cipher)
        opt  = cpu_opt_series(df, cipher)
        if base.empty or opt.empty:
            continue
        base_max = base["keys_per_sec"].max()
        opt_max  = opt["keys_per_sec"].max()
        if base_max > 0 and opt_max > 0:
            records.append((cd(cipher), opt_max / base_max))

    if not records:
        return
    records.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*records)

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(names, vals, color=CIPHER_COLOURS[:len(names)],
                  edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="1× (no improvement)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02*max(vals),
                f"{val:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("CPU Opt Speedup over CPU Baseline (×)", fontsize=11)
    ax.set_title("CPU Optimisation Speedup over Baseline — All Ciphers", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    path = os.path.join(outdir, "summary__cpu_opt_speedup.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  saved: {path}")


# ── summary: GPU speedup over CPU-opt ─────────────────────────────────────────

def summary_gpu_speedup_over_cpu_opt(df, ciphers, outdir):
    records = []
    for cipher in ciphers:
        opt = cpu_opt_series(df, cipher)
        gpu = best_gpu(df, cipher)
        if opt.empty or gpu.empty:
            continue
        opt_max = opt["keys_per_sec"].max()
        gpu_max = gpu["keys_per_sec"].max()
        if opt_max > 0 and gpu_max > 0:
            records.append((cd(cipher), gpu_max / opt_max))

    if not records:
        return
    records.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*records)

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(names, vals, color=CIPHER_COLOURS[:len(names)],
                  edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="1× (GPU = CPU Opt)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02*max(vals),
                f"{val:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("GPU Speedup over CPU Optimised (×)", fontsize=11)
    ax.set_title("Best GPU Speedup over CPU Optimised — All Ciphers", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    path = os.path.join(outdir, "summary__gpu_speedup_over_cpu_opt.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  saved: {path}")


# ── summary: all-cipher throughput overlay ────────────────────────────────────

def summary_all_throughput_overlay(df, ciphers, outdir):
    """
    Two subplots side by side:
      Left:  CPU Optimised — throughput vs bits, one line per cipher
      Right: Best GPU      — same
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, role, title_suffix in [
        (axes[0], "cpu_opt", "CPU Optimised"),
        (axes[1], "gpu",     "Best GPU"),
    ]:
        for idx, cipher in enumerate(ciphers):
            if role == "cpu_opt":
                s = cpu_opt_series(df, cipher)
            else:
                s = best_gpu(df, cipher)
            if s.empty:
                continue
            ax.plot(s["unknown_bits"], s["keys_per_sec"],
                    marker="o", markersize=3, linewidth=1.5,
                    color=CIPHER_COLOURS[idx % len(CIPHER_COLOURS)],
                    label=cd(cipher))
        ax.set_yscale("log")
        ax.set_title(f"Throughput vs Bits — {title_suffix}", fontsize=12)
        ax.set_xlabel("Unknown key bits", fontsize=10)
        ax.set_ylabel("keys/s (log)", fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("CPU Optimised vs Best GPU Throughput — All Ciphers", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, "summary__all_throughput_overlay.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {path}")


# ── summary: head-to-head speedup lines over bits ─────────────────────────────

def summary_speedup_vs_bits(df, ciphers, outdir):
    """
    Line plot: GPU/CPU-opt speedup ratio vs bits for every cipher.
    Shows how GPU advantage changes as key-space grows.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, cipher in enumerate(ciphers):
        opt = cpu_opt_series(df, cipher)
        gpu = best_gpu(df, cipher)
        if opt.empty or gpu.empty:
            continue
        # Align on common bit values
        merged = pd.merge(
            opt[["unknown_bits","keys_per_sec"]].rename(columns={"keys_per_sec":"cpu_kps"}),
            gpu[["unknown_bits","keys_per_sec"]].rename(columns={"keys_per_sec":"gpu_kps"}),
            on="unknown_bits"
        )
        if merged.empty:
            continue
        merged["speedup"] = merged["gpu_kps"] / merged["cpu_kps"]
        ax.plot(merged["unknown_bits"], merged["speedup"],
                marker="o", markersize=3, linewidth=1.5,
                color=CIPHER_COLOURS[idx % len(CIPHER_COLOURS)],
                label=cd(cipher))

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="1× (parity)")
    ax.set_title("GPU / CPU-Opt Speedup Ratio vs Unknown Bits — All Ciphers", fontsize=13)
    ax.set_xlabel("Unknown key bits", fontsize=11)
    ax.set_ylabel("GPU speedup over CPU Optimised (×)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    path = os.path.join(outdir, "summary__speedup_vs_bits.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Plot CPU-opt vs GPU benchmark results")
    ap.add_argument("--cpu_csv",     required=True, help="CPU optimisation results CSV")
    ap.add_argument("--gpu_csv",     required=True, help="GPU / baseline results CSV")
    ap.add_argument("--outdir",      default="plots_cpu_vs_gpu", help="Output directory")
    ap.add_argument("--only_cipher", default=None, help="Restrict to one cipher")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    print(f"Loading {args.cpu_csv} ...")
    print(f"Loading {args.gpu_csv} ...")
    df = load_csvs(args.cpu_csv, args.gpu_csv)

    all_ciphers = sorted(df["cipher"].unique())
    ciphers = [args.only_cipher] if args.only_cipher else all_ciphers

    if args.only_cipher and args.only_cipher not in all_ciphers:
        raise SystemExit(f"Cipher '{args.only_cipher}' not found. Available: {all_ciphers}")

    print(f"\nGenerating per-cipher plots for: {ciphers}")
    for cipher in ciphers:
        per_cipher_throughput(df, cipher, args.outdir)
        per_cipher_time(df, cipher, args.outdir)

    if not args.only_cipher:
        print("\nGenerating summary plots ...")
        summary_throughput_comparison(df, ciphers, args.outdir)
        summary_cpu_opt_speedup(df, ciphers, args.outdir)
        summary_gpu_speedup_over_cpu_opt(df, ciphers, args.outdir)
        summary_all_throughput_overlay(df, ciphers, args.outdir)
        summary_speedup_vs_bits(df, ciphers, args.outdir)

    print(f"\nAll plots saved to: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()

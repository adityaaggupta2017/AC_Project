#!/usr/bin/env python3
"""
plot_results.py  —  Visualise brute-force benchmark results.

Generates per-cipher plots PLUS cross-cipher summary plots:
  <cipher>__time_vs_bits.png          — wall-clock time vs unknown_bits (log y)
  <cipher>__throughput_vs_bits.png    — keys/s vs unknown_bits (log y)
  summary__best_gpu_throughput.png    — best GPU keys/s for every cipher at max bits
  summary__cpu_vs_gpu_speedup.png     — GPU speedup factor over CPU per cipher
  summary__all_throughput.png         — throughput vs bits for all ciphers (best variant each)
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── colour palette (12 distinct colours) ────────────────────────────────────
CIPHER_COLOURS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78",
]

PLATFORM_STYLE = {
    "cpu": dict(linestyle="--", marker="s", linewidth=1.4),
    "gpu": dict(linestyle="-",  marker="o", linewidth=1.8),
    "extrapolation": None,   # skip in regular plots
}

CIPHER_DISPLAY = {
    "simon32_64":    "SIMON 32/64",
    "present80":     "PRESENT-80",
    "speck64_128":   "SPECK 64/128",
    "grain_v1":      "Grain v1",
    "trivium":       "Trivium",
    "chacha20":      "ChaCha20",
    "tinyjambu_128": "TinyJAMBU-128",
    "zuc_128":       "ZUC-128",
    "snow_v":        "SNOW-V",
    "aes128":        "AES-128",
    "salsa20":       "Salsa20",
    "grain128aeadv2":"Grain-128AEADv2",
}

def safe_label(s):
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def cipher_display(name):
    return CIPHER_DISPLAY.get(name, name)

# ── per-cipher plots ─────────────────────────────────────────────────────────

def plot_time_vs_bits(df, cipher, outdir):
    d = df[(df["cipher"] == cipher) & (df["unknown_bits"] > 0)].copy()
    if d.empty:
        return
    d = d.sort_values(["platform", "variant", "unknown_bits"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (platform, variant), g in d.groupby(["platform", "variant"], sort=False):
        if platform == "extrapolation":
            continue
        style = PLATFORM_STYLE.get(platform, {})
        if style is None:
            continue
        g = g.sort_values("unknown_bits")
        ax.plot(g["unknown_bits"], g["seconds"],
                label=f"{platform}:{variant}", **style)

    ax.set_yscale("log")
    ax.set_title(f"{cipher_display(cipher)} — Attack Time vs Unknown Bits", fontsize=13)
    ax.set_xlabel("Unknown key bits (u)", fontsize=11)
    ax.set_ylabel("Time (seconds, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(outdir, f"{safe_label(cipher)}__time_vs_bits.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_throughput_vs_bits(df, cipher, outdir):
    d = df[(df["cipher"] == cipher) & (df["unknown_bits"] > 0)].copy()
    if d.empty:
        return
    d = d.sort_values(["platform", "variant", "unknown_bits"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (platform, variant), g in d.groupby(["platform", "variant"], sort=False):
        if platform == "extrapolation":
            continue
        style = PLATFORM_STYLE.get(platform, {})
        if style is None:
            continue
        g = g.sort_values("unknown_bits")
        ax.plot(g["unknown_bits"], g["keys_per_sec"],
                label=f"{platform}:{variant}", **style)

    ax.set_yscale("log")
    ax.set_title(f"{cipher_display(cipher)} — Throughput vs Unknown Bits", fontsize=13)
    ax.set_xlabel("Unknown key bits (u)", fontsize=11)
    ax.set_ylabel("Throughput (keys/s, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(outdir, f"{safe_label(cipher)}__throughput_vs_bits.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path}")


# ── summary plots ─────────────────────────────────────────────────────────────

def plot_summary_best_gpu_throughput(df, ciphers, outdir):
    """Bar chart: best GPU keys/s per cipher at the largest measured bit count."""
    records = []
    for cipher in ciphers:
        g = df[(df["cipher"] == cipher) & (df["platform"] == "gpu") & (df["unknown_bits"] > 0)]
        if g.empty:
            continue
        max_bits = g["unknown_bits"].max()
        best = g[g["unknown_bits"] == max_bits]["keys_per_sec"].max()
        records.append((cipher_display(cipher), best))
    if not records:
        return

    records.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*records)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, vals, color=CIPHER_COLOURS[:len(names)], edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Best GPU Throughput (keys/s, log scale)", fontsize=11)
    ax.set_title("Best GPU Throughput Comparison — All Ciphers", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    # annotate bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                f"{val:.2e}", ha="center", va="bottom", fontsize=7)
    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    path = os.path.join(outdir, "summary__best_gpu_throughput.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_summary_cpu_vs_gpu_speedup(df, ciphers, outdir):
    """Bar chart: best GPU / CPU throughput speedup factor per cipher."""
    records = []
    for cipher in ciphers:
        sub = df[(df["cipher"] == cipher) & (df["unknown_bits"] > 0)]
        cpu_max = sub[sub["platform"] == "cpu"]["keys_per_sec"].max()
        gpu_max = sub[sub["platform"] == "gpu"]["keys_per_sec"].max()
        if cpu_max > 0 and gpu_max > 0:
            records.append((cipher_display(cipher), gpu_max / cpu_max))
    if not records:
        return

    records.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*records)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, vals, color=CIPHER_COLOURS[:len(names)], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("GPU Speedup over CPU (×)", fontsize=11)
    ax.set_title("GPU Speedup over CPU — All Ciphers (Best Variant)", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="1× (no speedup)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    path = os.path.join(outdir, "summary__cpu_vs_gpu_speedup.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_summary_all_throughput(df, ciphers, outdir):
    """Line plot: best-variant throughput vs bits for all ciphers on one axes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, cipher in enumerate(ciphers):
        g = df[(df["cipher"] == cipher) & (df["platform"] == "gpu") & (df["unknown_bits"] > 0)]
        if g.empty:
            continue
        # pick best throughput per bit value across all GPU variants
        best = g.groupby("unknown_bits")["keys_per_sec"].max().reset_index()
        best = best.sort_values("unknown_bits")
        ax.plot(best["unknown_bits"], best["keys_per_sec"],
                marker="o", markersize=3, linewidth=1.6,
                color=CIPHER_COLOURS[idx % len(CIPHER_COLOURS)],
                label=cipher_display(cipher))

    ax.set_yscale("log")
    ax.set_title("GPU Throughput vs Unknown Bits — All Ciphers (Best Variant)", fontsize=13)
    ax.set_xlabel("Unknown key bits (u)", fontsize=11)
    ax.set_ylabel("Throughput (keys/s, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    path = os.path.join(outdir, "summary__all_gpu_throughput.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_summary_all_time(df, ciphers, outdir):
    """Line plot: best-variant attack time vs bits for all ciphers on one axes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, cipher in enumerate(ciphers):
        g = df[(df["cipher"] == cipher) & (df["platform"] == "gpu") & (df["unknown_bits"] > 0)]
        if g.empty:
            continue
        best = g.groupby("unknown_bits")["seconds"].min().reset_index()
        best = best.sort_values("unknown_bits")
        ax.plot(best["unknown_bits"], best["seconds"],
                marker="o", markersize=3, linewidth=1.6,
                color=CIPHER_COLOURS[idx % len(CIPHER_COLOURS)],
                label=cipher_display(cipher))

    ax.set_yscale("log")
    ax.set_title("GPU Attack Time vs Unknown Bits — All Ciphers (Best Variant)", fontsize=13)
    ax.set_xlabel("Unknown key bits (u)", fontsize=11)
    ax.set_ylabel("Attack time (seconds, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    path = os.path.join(outdir, "summary__all_gpu_time.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Plot brute-force benchmark results")
    ap.add_argument("csv",          nargs="?",       help="Path to results CSV")
    ap.add_argument("--csv",        dest="csv_flag", help="Path to results CSV")
    ap.add_argument("--outdir",     default="plots", help="Output directory for images")
    ap.add_argument("--only_cipher",default=None,    help="Plot only this cipher")
    args = ap.parse_args()
    csv_path = args.csv_flag if args.csv_flag else args.csv
    if not csv_path:
        raise SystemExit("Provide the CSV file: python3 plot_results.py results.csv")
    return csv_path, args.outdir, args.only_cipher


def main():
    csv_path, outdir, only_cipher = parse_args()
    ensure_dir(outdir)

    df = pd.read_csv(csv_path)
    needed = ["cipher", "platform", "variant", "unknown_bits", "seconds", "keys_per_sec"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    df["unknown_bits"] = pd.to_numeric(df["unknown_bits"], errors="coerce")
    df["seconds"]      = pd.to_numeric(df["seconds"],      errors="coerce")
    df["keys_per_sec"] = pd.to_numeric(df["keys_per_sec"], errors="coerce")
    df = df.dropna(subset=["unknown_bits", "seconds", "keys_per_sec"])

    all_ciphers = sorted([c for c in df["cipher"].unique() if c != "cipher"])

    if only_cipher:
        if only_cipher not in all_ciphers:
            raise SystemExit(f"Cipher '{only_cipher}' not found. Available: {all_ciphers}")
        ciphers = [only_cipher]
    else:
        ciphers = all_ciphers

    print(f"\nGenerating per-cipher plots for: {ciphers}")
    for cipher in ciphers:
        plot_time_vs_bits(df, cipher, outdir)
        plot_throughput_vs_bits(df, cipher, outdir)

    if not only_cipher:
        print("\nGenerating summary plots...")
        plot_summary_best_gpu_throughput(df, ciphers, outdir)
        plot_summary_cpu_vs_gpu_speedup(df, ciphers, outdir)
        plot_summary_all_throughput(df, ciphers, outdir)
        plot_summary_all_time(df, ciphers, outdir)

    print(f"\nAll plots saved to: {os.path.abspath(outdir)}/")


if __name__ == "__main__":
    main()

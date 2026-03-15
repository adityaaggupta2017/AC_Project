#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def safe_label(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="Path to results CSV")
    ap.add_argument("--csv", dest="csv_flag", help="Path to results CSV")
    ap.add_argument("--outdir", default="plots", help="Output directory for images")
    ap.add_argument("--only_cipher", default=None, help="Optional single cipher name")
    args = ap.parse_args()
    csv_path = args.csv_flag if args.csv_flag else args.csv
    if not csv_path:
        raise SystemExit("Please provide the CSV file as --csv file.csv or as a positional argument.")
    return csv_path, args.outdir, args.only_cipher

def plot_time_vs_bits(df, cipher, outdir):
    d = df[df["cipher"] == cipher].copy().sort_values(["unknown_bits","platform","variant"])
    fig = plt.figure()
    ax = plt.gca()
    for (platform, variant), g in d.groupby(["platform", "variant"]):
        g = g.sort_values("unknown_bits")
        ax.plot(g["unknown_bits"], g["seconds"], marker="o", label=f"{platform}:{variant}")
    ax.set_title(f"{cipher} — time vs unknown_bits")
    ax.set_xlabel("unknown_bits")
    ax.set_ylabel("seconds")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{safe_label(cipher)}__time_vs_bits.png"), dpi=200)
    plt.close(fig)

def plot_throughput_vs_bits(df, cipher, outdir):
    d = df[df["cipher"] == cipher].copy().sort_values(["unknown_bits","platform","variant"])
    fig = plt.figure()
    ax = plt.gca()
    for (platform, variant), g in d.groupby(["platform", "variant"]):
        g = g.sort_values("unknown_bits")
        ax.plot(g["unknown_bits"], g["keys_per_sec"], marker="o", label=f"{platform}:{variant}")
    ax.set_title(f"{cipher} — throughput vs unknown_bits")
    ax.set_xlabel("unknown_bits")
    ax.set_ylabel("keys/sec")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{safe_label(cipher)}__throughput_vs_bits.png"), dpi=200)
    plt.close(fig)

def main():
    csv_path, outdir, only_cipher = parse_args()
    ensure_dir(outdir)
    df = pd.read_csv(csv_path)
    needed = ["cipher","platform","variant","unknown_bits","seconds","keys_per_sec"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    df["unknown_bits"] = df["unknown_bits"].astype(int)
    df["seconds"] = df["seconds"].astype(float)
    df["keys_per_sec"] = df["keys_per_sec"].astype(float)

    ciphers = sorted(df["cipher"].unique().tolist())
    if only_cipher:
        if only_cipher not in ciphers:
            raise SystemExit(f"Cipher '{only_cipher}' not found. Available: {ciphers}")
        ciphers = [only_cipher]

    for cipher in ciphers:
        plot_time_vs_bits(df, cipher, outdir)
        plot_throughput_vs_bits(df, cipher, outdir)

    print(f"Saved plots to: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()

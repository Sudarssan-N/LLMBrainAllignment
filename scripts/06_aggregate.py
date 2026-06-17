"""Stage 6: aggregate per-model results into one comparison table + tidy CSV.

Scans data/derived/<model>/ for encoding.json / varpart.json / rsa.json and summarizes the
key metrics for the model sweep: best-layer normalized predictivity, peak unique-hidden vs
unique-surprisal, best RSA. Writes data/derived/summary.csv and prints a table.

Usage:
    python scripts/06_aggregate.py
    python scripts/06_aggregate.py --models gpt2,pythia-160m,pythia-410m
"""

from __future__ import annotations

import argparse
import csv

import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.utils.io import load_json


def _peak(per_layer: dict, key: str):
    """Return (layer, value) maximizing per_layer[layer][key] (ignoring None)."""
    best_l, best_v = None, float("-inf")
    for l, d in per_layer.items():
        v = d.get(key) if isinstance(d, dict) else None
        if v is not None and v > best_v:
            best_l, best_v = int(l), v
    return best_l, (None if best_l is None else best_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=None, help="comma list; default = all dirs found")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = cfg.derived_dir()  # data/derived
    model_dirs = (
        [base / m for m in args.models.split(",")]
        if args.models
        else sorted(p for p in base.iterdir() if p.is_dir())
    )

    rows = []
    for d in model_dirs:
        row = {"model": d.name}
        enc = d / "encoding.json"
        if enc.exists():
            e = load_json(enc)
            row["best_layer"] = e.get("best_layer")
            row["best_norm_r"] = e.get("best_normalized_r")
            row["best_mean_r"] = e.get("best_mean_r")
            row["n_layers"] = len(e.get("per_layer", {}))
        vp = d / "varpart.json"
        if vp.exists():
            vj = load_json(vp)
            v = vj["per_layer"]
            # Prefer the Stage-4-designated peak layer (carries bootstrap CI); fall back to
            # a max search for older runs.
            l_uh = vj.get("peak_layer")
            l_uh = int(l_uh) if l_uh is not None else _peak(v, "mean_unique_hidden")[0]
            peak = v[str(l_uh)] if l_uh is not None else {}
            uh = peak.get("mean_unique_hidden")
            us = peak.get("mean_unique_surprisal")
            row["peak_unique_hidden"] = uh
            row["peak_unique_hidden_layer"] = l_uh
            row["unique_surprisal_at_peak"] = us
            if peak.get("ci_unique_hidden"):
                c = peak["ci_unique_hidden"]
                row["unique_hidden_ci"] = f"[{c['lo']:.4f},{c['hi']:.4f}]"
            if uh and us:
                row["hidden_over_surprisal"] = round(uh / us, 1)
        rsa = d / "rsa.json"
        if rsa.exists():
            r = load_json(rsa)
            row["best_rsa"] = r.get("best_rsa")
            row["best_rsa_layer"] = r.get("best_layer")
        rows.append(row)

    if not rows:
        print("[stage6] no results found under data/derived/")
        return

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
    out = base / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # pretty print
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in fields}
    print(" | ".join(k.ljust(widths[k]) for k in fields))
    print("-+-".join("-" * widths[k] for k in fields))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(widths[k]) for k in fields))
    print(f"\n[stage6] wrote {out}")


if __name__ == "__main__":
    main()

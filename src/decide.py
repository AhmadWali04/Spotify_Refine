"""M5: apply the PRD1 decision criteria to experiments.csv and pick the featurizer.

    python -m src.decide
    python -m src.decide --library data/raw/library_synthetic.json

Rules (PRD1, "Decision criteria"):
  1. Must beat baseline: top-1 >= 2x E0 top-1 on every fold.
  2. Must cover the library: >= 90% of liked songs, or a fallback method that covers the rest.
  3. Among passing runs, keep those whose top-3 is within one fold-std of the best top-3.
  4. Prefer simple: fewer sources, then fewer dims, then faster featurizing.
  5. Prefer explainable: Essentia beats CLAP on a tie.

Writes data/decision.json (read by the sorter) and reports/decision.md (the decision note).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.vectors import config_for_run, load_config

MIN_COVERAGE = 90.0


def _sources(cfg: dict) -> set[str]:
    """Which raw sources a config needs (tags / essentia / clap)."""
    m = cfg["method"]
    if m in ("concat", "concat_pca", "interactions"):
        return set().union(*(_sources({"method": b["method"]}) for b in cfg["params"]["blocks"]))
    if m.startswith("tags"):
        return {"tags"}
    return {m}


def load_runs(log: Path) -> pd.DataFrame:
    if not log.exists():
        raise FileNotFoundError(f"{log.name} not found; run experiments first.")
    df = pd.read_csv(log)
    df = df.sort_values("timestamp").groupby("run_id", as_index=False).last()   # latest row per run
    df["folds"] = df["top1_folds"].astype(str).str.split().apply(lambda v: np.array(v, dtype=float))
    return df


def decide(df: pd.DataFrame) -> dict:
    base = df[df["method"] == "random"]
    if base.empty:
        raise ValueError("No E0 (random) run in the log; run configs/e0_random.yaml first.")
    e0_folds = base.iloc[0]["folds"]

    rows, rejected = [], []
    for _, r in df[df["method"] != "random"].iterrows():
        try:
            cfg_path = config_for_run(r["run_id"])
        except FileNotFoundError:
            rejected.append((r["run_id"], "config file not found"))
            continue
        cfg = load_config(cfg_path)
        f = r["folds"]
        if len(f) != len(e0_folds) or not np.all(f >= 2 * e0_folds):
            rejected.append((r["run_id"], "top-1 not >= 2x E0 on every fold"))
            continue
        rows.append({**r.to_dict(), "config": str(cfg_path.relative_to(paths.ROOT)),
                     "sources": sorted(_sources(cfg)), "n_sources": len(_sources(cfg))})
    if not rows:
        raise ValueError("No run beats the baseline yet. Rejected: " + "; ".join(f"{a} ({b})" for a, b in rejected))

    passing = pd.DataFrame(rows)
    best = passing.loc[passing["top3_mean"].idxmax()]
    tied = passing[passing["top3_mean"] >= best["top3_mean"] - best["top3_std"]].copy()
    # Explainability: on a tie, an Essentia-only run is ranked as if it used fewer sources than CLAP.
    tied["explain"] = tied["sources"].apply(lambda s: 0 if "clap" not in s else 1)
    tied = tied.sort_values(["n_sources", "explain", "dims", "featurize_min", "top3_mean"],
                            ascending=[True, True, True, True, False])
    win = tied.iloc[0]

    fallback = None
    if win["coverage_liked"] < MIN_COVERAGE:
        # Cover the rest with the best passing run that reaches more songs (typically tags).
        others = passing[(passing["run_id"] != win["run_id"]) & (passing["coverage_liked"] > win["coverage_liked"])]
        if not others.empty:
            wide = others[others["coverage_liked"] >= MIN_COVERAGE]
            pick = (wide if not wide.empty else others).sort_values("top3_mean", ascending=False).iloc[0]
            fallback = {"run_id": pick["run_id"], "config": pick["config"],
                        "coverage_liked": float(pick["coverage_liked"]), "top3_mean": float(pick["top3_mean"])}

    def metrics(r):
        return {k: float(r[k]) for k in ("top1_mean", "top1_std", "top3_mean", "top3_std",
                                         "macro_f1_mean", "coverage_liked", "dims", "featurize_min")}

    return {
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": win["run_id"], "config": win["config"], "sources": win["sources"],
        "metrics": metrics(win),
        "best_top3_run": best["run_id"],
        "tied_with": [r for r in tied["run_id"] if r != win["run_id"]],
        "fallback": fallback,
        "coverage_ok": bool(win["coverage_liked"] >= MIN_COVERAGE or
                            (fallback and fallback["coverage_liked"] >= MIN_COVERAGE)),
        "e0_top1": float(base.iloc[0]["top1_mean"]),
        "rejected": [{"run_id": a, "reason": b} for a, b in rejected],
    }


def note(d: dict) -> str:
    m = d["metrics"]
    lines = [
        "# Decision note", "",
        f"_Generated {d['decided_at']} by `python -m src.decide`._", "",
        f"**Winner: {d['run_id']}** (`{d['config']}`, sources: {', '.join(d['sources'])})", "",
        f"- Top-1 {m['top1_mean']:.1f} ± {m['top1_std']:.1f}% (E0: {d['e0_top1']:.1f}%)",
        f"- Top-3 {m['top3_mean']:.1f} ± {m['top3_std']:.1f}%",
        f"- Macro-F1 {m['macro_f1_mean']:.3f}, {int(m['dims'])} dims, {m['featurize_min']:.1f} min to featurize",
        f"- Coverage {m['coverage_liked']:.1f}% of liked songs", "",
    ]
    if d["tied_with"]:
        lines.append(f"Within one std of the best top-3 ({d['best_top3_run']}): {', '.join(d['tied_with'])}. "
                     "The winner is the simplest of these (fewer sources, then smaller, then faster).")
    else:
        lines.append("The winner has the best top-3 outright.")
    if d["fallback"]:
        f = d["fallback"]
        lines.append(f"\nFallback for songs the winner can't vectorize: **{f['run_id']}** "
                     f"({f['coverage_liked']:.1f}% coverage, top-3 {f['top3_mean']:.1f}%).")
    if not d["coverage_ok"]:
        lines.append(f"\n**Warning:** coverage is under {MIN_COVERAGE:.0f}% even with the fallback.")
    if d["rejected"]:
        lines.append("\nRejected: " + "; ".join(f"{r['run_id']} ({r['reason']})" for r in d["rejected"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(paths.LIBRARY))
    args = ap.parse_args()
    lib_path = Path(args.library)
    suffix = paths.library_suffix(lib_path)
    d = decide(load_runs(paths.ROOT / f"experiments{suffix}.csv"))
    paths.write_json(paths.for_library(lib_path, "decision.json"), d)
    text = note(d)
    (paths.REPORTS / f"decision{suffix}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

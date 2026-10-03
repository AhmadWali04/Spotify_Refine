"""PCA and SVD plots of every song in a vector space (sorted and unsorted alike).

    python -m src.plots                                   # space from data/decision.json
    python -m src.plots --config configs/e1b_tags_lsa.yaml
    python -m src.plots --color tier                      # color by sorter tier (needs suggestions.json)
    python -m src.plots --color cluster                   # color leftover songs by new-playlist cluster
    python -m src.plots --library data/raw/library_synthetic.json

Writes to reports/plots/<run_id>/:
    pca_scatter.png    all songs on PC1 x PC2
    pca_pairs.png      every pair of PC1..PC4 (structure often hides past the first two)
    pca_variance.png   scree + cumulative explained variance (how many dims carry signal)
    svd_scatter.png    all songs on the first two right-singular directions, uncentered
    svd_spectrum.png   singular values, log scale

PCA vs SVD: PCA is the SVD of the *centered* matrix, so its axes describe how songs differ from
the average song. The plain SVD (what LSA uses) is uncentered, so its first direction mostly points
at the average song itself and the interesting contrasts start at the second. For z-scored spaces
the columns are already centered and the two agree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import paths  # noqa: E402
from src.linalg import pca, truncated_svd  # noqa: E402
from src.vectors import load_space  # noqa: E402

UNSORTED = "not in a playlist"
OTHER = "other playlists"
GREY = {UNSORTED: "#9a9da3", OTHER: "#d4d4d0", "no cluster": "#d4d4d0", "sorted": "#d4d4d0"}
TIER_COLORS = {"confident": "#1a8f4c", "suggested": "#2c5fb3", "leftover": "#d08300", "no_data": "#9a9da3",
               "sorted": "#d4d4d0"}


# ---------------------------------------------------------------- labels

def playlist_labels(lib: dict, ids: list[str], top_n: int) -> tuple[list[str], list[str]]:
    """Each song's label = its smallest (most specific) playlist; only the top_n biggest get a color."""
    size = {p["id"]: len(p["track_ids"]) for p in lib["playlists"]}
    name = {p["id"]: p["name"] for p in lib["playlists"]}
    member: dict[str, str] = {}
    for p in lib["playlists"]:
        for t in p["track_ids"]:
            if t not in member or size[p["id"]] < size[member[t]]:
                member[t] = p["id"]
    counts: dict[str, int] = {}
    for t in ids:
        if t in member:
            counts[member[t]] = counts.get(member[t], 0) + 1
    top = sorted(counts, key=lambda p: -counts[p])[:top_n]
    labels = [name[member[t]] if member.get(t) in top else (OTHER if t in member else UNSORTED) for t in ids]
    return labels, [name[p] for p in top]


def suggestion_labels(lib_path: Path, ids: list[str], by: str) -> tuple[list[str], list[str]]:
    sp = paths.for_library(lib_path, "suggestions.json")
    if not sp.exists():
        raise FileNotFoundError(f"--color {by} needs {sp.name}; run `python -m src.sorter.run` first.")
    sugg = json.loads(sp.read_text())
    songs = {s["track_id"]: s for s in sugg["songs"]}
    if by == "tier":
        labels = [songs[t]["tier"] if t in songs else "sorted" for t in ids]
        return labels, [k for k in TIER_COLORS if k != "sorted"]
    names = {c["id"]: c["name"] for c in sugg["clusters"]}
    labels = [names.get(songs[t]["cluster"], "no cluster") if t in songs else OTHER for t in ids]
    return labels, list(names.values())


def _colors(order: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    out = {lab: TIER_COLORS.get(lab) or matplotlib.colors.to_hex(cmap(i % cmap.N)) for i, lab in enumerate(order)}
    return {**GREY, **out}


def _scatter(ax, Z: np.ndarray, labels: np.ndarray, order: list[str], colors: dict, legend: bool = True):
    """Grey background groups first, highlighted groups on top."""
    background = [g for g in (OTHER, "no cluster", "sorted", UNSORTED) if g in set(labels) and g not in order]
    for g in background + order:
        m = labels == g
        if not m.any():
            continue
        highlighted = g in order
        ax.scatter(Z[m, 0], Z[m, 1], s=9 if highlighted else 5, color=colors[g], alpha=0.85 if highlighted else 0.45,
                   linewidths=0, label=f"{g[:30]} ({m.sum()})", rasterized=True)
    if legend:
        ax.legend(fontsize=7, markerscale=2, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)


# ---------------------------------------------------------------- plots

def make_plots(X: np.ndarray, labels: list[str], order: list[str], title: str, out_dir: Path,
               n_components: int = 50) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=object)
    colors = _colors(order)
    k = min(n_components, *X.shape)
    written = []

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # PCA: scores on all available components; explained-variance ratio is relative to *total* variance.
    Z, _, evr = pca(X, k)

    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter(ax, Z[:, :2], labels, order, colors)
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% of variance)" if k > 1 else "")
    ax.set_title(f"{title}: PCA of {len(X)} songs")
    save(fig, "pca_scatter.png")

    m = min(4, Z.shape[1])
    if m >= 3:
        fig, axes = plt.subplots(m - 1, m - 1, figsize=(3.4 * (m - 1) + 2.5, 3.2 * (m - 1)), squeeze=False)
        for i in range(1, m):
            for j in range(m - 1):
                ax = axes[i - 1, j]
                if j >= i:
                    ax.axis("off")
                    continue
                _scatter(ax, Z[:, [j, i]], labels, order, colors, legend=False)
                ax.tick_params(labelsize=6)
                if i == m - 1:
                    ax.set_xlabel(f"PC{j + 1} ({100 * evr[j]:.1f}%)", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"PC{i + 1} ({100 * evr[i]:.1f}%)", fontsize=8)
        handles, labs = axes[-1, 0].get_legend_handles_labels()
        fig.legend(handles, labs, fontsize=7, markerscale=2, loc="upper right", frameon=False)
        fig.suptitle(f"{title}: PCA component pairs")
        save(fig, "pca_pairs.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(1, len(evr) + 1)
    ax.bar(idx, 100 * evr, color="#2c5fb3", alpha=0.75, label="per component")
    ax2 = ax.twinx()
    cum = 100 * np.cumsum(evr)
    ax2.plot(idx, cum, color="#1a8f4c", marker="o", ms=3, label="cumulative")
    for target in (50, 80, 90):
        hit = np.searchsorted(cum, target)
        if hit < len(cum):
            ax2.axhline(target, color="#9a9da3", lw=0.6, ls="--")
            ax2.annotate(f"{target}% at {hit + 1} PCs", (hit + 1, target), fontsize=7, xytext=(4, -10),
                         textcoords="offset points")
    ax.set_xlabel("principal component")
    ax.set_ylabel("% of variance")
    ax2.set_ylabel("cumulative %")
    ax2.set_ylim(0, 102)
    ax.set_title(f"{title}: explained variance (first {len(evr)} of {min(X.shape)} PCs)")
    save(fig, "pca_variance.png")

    # SVD: uncentered, as LSA uses it.
    U_S, _, S = truncated_svd(X, k)
    fig, ax = plt.subplots(figsize=(10, 7))
    if U_S.shape[1] >= 2:
        _scatter(ax, U_S[:, :2], labels, order, colors)
        ax.set_xlabel(f"SV1 (σ = {S[0]:.3g})")
        ax.set_ylabel(f"SV2 (σ = {S[1]:.3g})")
    ax.set_title(f"{title}: SVD (uncentered) of {len(X)} songs")
    save(fig, "svd_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(np.arange(1, len(S) + 1), S, color="#2c5fb3", marker="o", ms=3)
    energy = 100 * S ** 2 / np.sum(X.astype(float) ** 2)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ (log scale)")
    ax.set_title(f"{title}: singular values (top {len(S)} hold {energy.sum():.0f}% of ‖X‖²)")
    save(fig, "svd_spectrum.png")
    return written


def main() -> None:
    from src.sorter import spaces

    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(paths.LIBRARY))
    ap.add_argument("--config", help="featurizer config (default: data/decision.json, or the demo config)")
    ap.add_argument("--color", choices=["playlist", "tier", "cluster"], default="playlist")
    ap.add_argument("--top", type=int, default=12, help="playlists to color individually (rest are grey)")
    ap.add_argument("--components", type=int, default=50, help="components shown in the variance and SVD plots")
    ap.add_argument("--liked-only", action="store_true", help="plot liked songs only, not every playlist track")
    args = ap.parse_args()

    lib_path = Path(args.library)
    lib = paths.load_library(lib_path)
    cfg = spaces.resolve(lib, lib_path, args.config)[0]
    ids, X = load_space(cfg, lib, lib_path)
    if args.liked_only:
        liked = {x["id"] for x in lib["liked"]}
        keep = [i for i, t in enumerate(ids) if t in liked]
        ids, X = [ids[i] for i in keep], X[keep]
    if len(ids) < 3:
        raise SystemExit(f"Only {len(ids)} songs have vectors in {cfg['run_id']}; nothing to plot.")

    if args.color == "playlist":
        labels, order = playlist_labels(lib, ids, args.top)
    else:
        labels, order = suggestion_labels(lib_path, ids, args.color)

    run_id = cfg["run_id"] + paths.library_suffix(lib_path)
    out = paths.REPORTS / "plots" / (run_id if args.color == "playlist" else f"{run_id}_{args.color}")
    for p in make_plots(X, labels, order, cfg["run_id"], out, args.components):
        print(f"Saved {p.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()

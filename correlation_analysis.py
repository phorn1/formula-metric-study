"""Analyse the correlation between automated metrics and human scores."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from collections import Counter, defaultdict
from itertools import combinations
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

DATA_PATH = Path(__file__).parent / "all_formulas.json"
OUTPUT_DIR = Path(__file__).parent / "correlation_plots"


def load_data(path: Path) -> list[dict]:
    """Load all_formulas.json and return rows with human scores."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        row
        for row in data
        if "human_study_scores" in row and row["human_study_scores"]
    ]


def extract_metric_vectors(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {metric_name: (metric_values, human_values)} for all metrics."""
    human_arr = np.array([np.mean(r["human_study_scores"]) for r in rows])

    def metric(key):
        """Extract a single metric and scale from 0-1 to 0-10."""
        return np.array([r["metrics"][key] for r in rows]) * 10

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "BLEU": (metric("bleu_score"), human_arr),
        "Levenshtein": (metric("levenshtein_similarity"), human_arr),
        "CDM": (metric("cdm_score"), human_arr),
    }

    llm_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for entry in row["llm_scores"]:
            llm_scores[entry["judge_model"]].append(entry["score"])

    for model, scores in sorted(llm_scores.items()):
        label = model.split("/")[-1] if "/" in model else model
        result[f"LLM: {label}"] = (np.array(scores), human_arr)

    return result


def compute_correlations(
    metric_vals: np.ndarray, human_vals: np.ndarray
) -> dict[str, tuple[float, float]]:
    pearson_r, pearson_p = pearsonr(metric_vals, human_vals)
    spearman_r, spearman_p = spearmanr(metric_vals, human_vals)
    kendall_t, kendall_p = kendalltau(metric_vals, human_vals)
    return {
        "Pearson r": (pearson_r, pearson_p),
        "Spearman ρ": (spearman_r, spearman_p),
        "Kendall τ": (kendall_t, kendall_p),
    }


def plot_scatter(ax, title, metric_vals, human_vals, color, metrics_text):
    """Plot a bubble scatter subplot."""
    plot_metrics = np.round(metric_vals).astype(int).clip(0, 10)
    plot_humans = np.round(human_vals).astype(int).clip(0, 10)
    counts = Counter(zip(plot_humans.tolist(), plot_metrics.tolist()))

    # Plot points with size coding
    for (x, y), count in counts.items():
        size = 50 + count * 30
        alpha = min(0.8, 0.4 + count * 0.1)
        ax.scatter(x, y, s=size, alpha=alpha, color=color,
                   edgecolors='white', linewidth=1.5)
        if count > 1:
            ax.annotate(str(count), (x, y), ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')

    # Perfect agreement line
    ax.plot([0, 10], [0, 10], 'r--', alpha=0.8, linewidth=2, label='Perfect Agreement')
    ax.set_xlabel('Human Scores (avg)')
    ax.set_ylabel(title)
    ax.set_title(f'{title} vs Human Scores')
    ax.grid(True, alpha=0.3)

    legend = ax.legend(loc='lower center', frameon=True, fancybox=True,
                       shadow=True, fontsize=8)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))
    ax.set_aspect('equal')

    ax.text(0.915, 0.05, metrics_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                      edgecolor='black', linewidth=1))


def make_figure(metrics_data, all_correlations, specs, output_path):
    """Create a 2-column scatter figure.

    specs: list of (metric_key, display_label, color)
    """
    n = len(specs)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 5 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for i, (key, label, color) in enumerate(specs):
        m_vals, h_vals = metrics_data[key]
        corrs = all_correlations[key]
        metrics_text = (f'Corr: {corrs["Pearson r"][0]:.3f}\n'
                        f'Spearman: {corrs["Spearman ρ"][0]:.3f}\n'
                        f'Kendall: {corrs["Kendall τ"][0]:.3f}')
        plot_scatter(axes_flat[i], label, m_vals, h_vals, color, metrics_text)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")


def main():
    rows = load_data(DATA_PATH)
    print(f"Loaded {len(rows)} rows with human_study_scores")

    # Inter-annotator agreement (rows with >= 2 human scores)
    multi = [r["human_study_scores"] for r in rows if len(r["human_study_scores"]) >= 2]
    if multi:
        max_ann = max(len(s) for s in multi)
        print(f"\nInter-annotator agreement ({len(multi)} rows with ≥2 scores, up to {max_ann} annotators):")

        # Krippendorff's alpha (handles missing data naturally)
        reliability_data = np.full((max_ann, len(multi)), np.nan)
        for col, scores in enumerate(multi):
            for row, score in enumerate(scores):
                reliability_data[row, col] = score
        alpha = krippendorff.alpha(reliability_data, level_of_measurement="interval")
        print(f"  Krippendorff's α (interval): {alpha:.4f}")

        # Pairwise agreement for all annotator pairs
        pair_results = []
        for i, j in combinations(range(max_ann), 2):
            valid = [(s[i], s[j]) for s in multi if len(s) > max(i, j)]
            if len(valid) < 3:
                continue
            hi = np.array([v[0] for v in valid])
            hj = np.array([v[1] for v in valid])
            pr, _ = pearsonr(hi, hj)
            sr, _ = spearmanr(hi, hj)
            kt, _ = kendalltau(hi, hj)
            mad = np.mean(np.abs(hi - hj))
            pair_results.append((i, j, len(valid), pr, sr, kt, mad))
            print(f"  Annotator {i+1} vs {j+1} (n={len(valid)}): "
                  f"Pearson={pr:.4f}, Spearman={sr:.4f}, Kendall={kt:.4f}, MAD={mad:.2f}")

        if len(pair_results) > 1:
            avg_pr = np.mean([r[3] for r in pair_results])
            avg_sr = np.mean([r[4] for r in pair_results])
            avg_kt = np.mean([r[5] for r in pair_results])
            avg_mad = np.mean([r[6] for r in pair_results])
            print(f"  Average pairwise:            "
                  f"Pearson={avg_pr:.4f}, Spearman={avg_sr:.4f}, Kendall={avg_kt:.4f}, MAD={avg_mad:.2f}")

        # Human ceiling: leave-one-out r(H_i, mean(H_{-i}))
        print(f"\n  Human ceiling — r(annotator, mean of others):")
        loo_pearsons, loo_spearmans, loo_kendalls = [], [], []
        for i in range(max_ann):
            rows_for_i = [s for s in multi if len(s) > i]
            if len(rows_for_i) < 3:
                continue
            hi = np.array([s[i] for s in rows_for_i])
            h_others = np.array([np.mean([s[j] for j in range(len(s)) if j != i])
                                 for s in rows_for_i])
            pr, _ = pearsonr(hi, h_others)
            sr, _ = spearmanr(hi, h_others)
            kt, _ = kendalltau(hi, h_others)
            loo_pearsons.append(pr)
            loo_spearmans.append(sr)
            loo_kendalls.append(kt)
            print(f"    Annotator {i+1} vs mean(others) (n={len(rows_for_i)}): "
                  f"Pearson={pr:.4f}, Spearman={sr:.4f}, Kendall={kt:.4f}")
        if loo_pearsons:
            print(f"    Average:                           "
                  f"Pearson={np.mean(loo_pearsons):.4f}, "
                  f"Spearman={np.mean(loo_spearmans):.4f}, "
                  f"Kendall={np.mean(loo_kendalls):.4f}")

    metrics = extract_metric_vectors(rows)

    # Compute correlations
    all_correlations = {}
    for name, (m_vals, h_vals) in metrics.items():
        all_correlations[name] = compute_correlations(m_vals, h_vals)

    # Print correlation table as Markdown
    print()
    print("| Metric | Pearson r | Spearman ρ | Kendall τ |")
    print("|--------|----------:|-----------:|----------:|")
    for name, corrs in all_correlations.items():
        pr = corrs['Pearson r'][0]
        sr = corrs['Spearman ρ'][0]
        kt = corrs['Kendall τ'][0]
        print(f"| {name} | {pr:.3f} | {sr:.3f} | {kt:.3f} |")

    # Plot
    OUTPUT_DIR.mkdir(exist_ok=True)
    palette = sns.color_palette("tab10")

    # All metrics
    all_specs = [(name, name, palette[i % len(palette)])
                 for i, name in enumerate(metrics)]
    make_figure(metrics, all_correlations, all_specs,
                OUTPUT_DIR / "correlation_all.png")

    # Paper figure (selected metrics)
    paper_specs = [
        ("BLEU", "BLEU (Rule-based)", palette[0]),
        ("LLM: gemini-3.1-flash-lite-preview", "Gemini-3.1-Flash-Lite (LLM-as-a-Judge)", palette[1]),
        ("Levenshtein", "Levenshtein (Rule-based)", palette[2]),
        ("LLM: gemini-3-flash-preview", "Gemini-3-Flash (LLM-as-a-Judge)", palette[3]),
        ("CDM", "CDM (Rule-based)", palette[4]),
        ("LLM: gpt-5-mini", "GPT-5-Mini (LLM-as-a-Judge)", palette[6]),
    ]
    make_figure(metrics, all_correlations, paper_specs,
                OUTPUT_DIR / "correlation.png")


if __name__ == "__main__":
    main()
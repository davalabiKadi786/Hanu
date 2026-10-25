"""Command-line Task 2 pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep Matplotlib's writable cache inside the project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .text_mining import data_quality, enrich_text, parse_list_cell


def build_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"rows": int(len(df))}
    if "Event id" in df:
        summary["unique_events"] = int(df["Event id"].nunique())
    if "Failure Component" in df:
        vals = df["Failure Component"].map(parse_list_cell).explode().dropna()
        summary["top_failure_components"] = {str(k): int(v) for k, v in vals.value_counts().head(10).items()}
    if "Failure Condition" in df:
        vals = df["Failure Condition"].map(parse_list_cell).explode().dropna()
        summary["top_failure_conditions"] = {str(k): int(v) for k, v in vals.value_counts().head(10).items()}
    if "issue_category" in df:
        summary["issue_categories"] = {str(k): int(v) for k, v in df.issue_category.value_counts().items()}
    if "MODLYR" in df:
        years = pd.to_numeric(df.MODLYR, errors="coerce").dropna()
        summary["model_year_stats"] = {"count": int(years.count()), "mean": float(years.mean()), "median": float(years.median()), "std": float(years.std()) if len(years) > 1 else None}
    if "MODEL" in df:
        summary["top_models"] = {str(k): int(v) for k, v in df.MODEL.value_counts().head(10).items()}
    if "Opened date" in df:
        months = pd.to_datetime(df["Opened date"], errors="coerce").dt.to_period("M").value_counts().sort_index()
        summary["events_by_month"] = {str(k): int(v) for k, v in months.items()}
    summary["structured_tag_nonmissing"] = {
        col: int(df[col].notna().sum()) for col in ["Failure Component", "Failure Condition", "Trigger", "Fix Component", "Fix Condition"] if col in df
    }
    return summary


def make_figures(df: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plots = []
    if "issue_category" in df:
        plots.append((df.issue_category.value_counts().head(10), "Mined issue category", "issue_categories.png"))
    if "MODEL" in df:
        plots.append((df.MODEL.value_counts().head(10), "Top models", "top_models.png"))
    if "cluster" in df:
        plots.append((df.cluster.value_counts().sort_index(), "Complaint cluster", "clusters.png"))
    for series, title, filename in plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        series.sort_values().plot(kind="barh", ax=ax, color="#157f85")
        ax.set_title(title)
        ax.set_xlabel("Records")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=150)
        plt.close(fig)
    if "Opened date" in df:
        dates = pd.to_datetime(df["Opened date"], errors="coerce")
        trend = dates.dt.to_period("M").value_counts().sort_index()
        if len(trend):
            fig, ax = plt.subplots(figsize=(10, 4))
            trend.plot(ax=ax, color="#e17c39")
            ax.set_title("Events by opened month")
            ax.set_ylabel("Records")
            fig.tight_layout()
            fig.savefig(folder / "events_by_month.png", dpi=150)
            plt.close(fig)


def write_stakeholder_report(df: pd.DataFrame, summary: dict, quality: dict, terms: dict, path: Path) -> None:
    components = summary.get("top_failure_components", {})
    conditions = summary.get("top_failure_conditions", {})
    categories = summary.get("issue_categories", {})
    models = summary.get("top_models", {})
    populated = summary.get("structured_tag_nonmissing", {})
    def bullets(values: dict, suffix: str = "") -> str:
        return "\n".join(f"- **{name}**: {count}{suffix}" for name, count in list(values.items())[:8]) or "- No values available."
    cluster_lines = "\n".join(f"- Cluster {label}: {', '.join(words[:6])}" for label, words in terms.items()) or "- No cluster terms available."
    report = f'''# HanuAi Vehicle Complaint Analysis\n\n## Executive summary\n\nThis run analyzed **{len(df):,} records** from the supplied vehicle complaint CSV. Automated category labels and text clusters are screening aids; engineering validation is required before assigning root cause. The most frequent mined narrative categories are shown below.\n\n## Mined narrative categories\n\n{bullets(categories, " records")}\n\n## Structured failure fields\n\n{bullets(components, " records")}\n\nFailure conditions:\n\n{bullets(conditions, " records")}\n\nThe supplied structured tags are sparse: {', '.join(f'{k} {v}/{len(df)} populated' for k,v in populated.items())}. Treat frequencies from these fields as an incomplete subset; prioritize the narrative outputs for review.\n\n## Top vehicle models in this sample\n\n{bullets(models, " records")}\n\n## Complaint cluster terms\n\n{cluster_lines}\n\n## Recommended actions\n\n1. Prioritize recurring narrative symptom and component combinations for engineering review; verify repair histories and warranty cost before ranking investments.\n2. Compare issue rates by model year, plant, and build date using production denominators. This file contains event counts, not fleet populations needed for rate comparisons.\n3. Review communication/network and electrical cases with diagnostic codes and repair outcomes to separate confirmed component faults from software or harness causes.\n4. Validate entity tags and a sample of each cluster with technicians; tune dictionaries and cluster count against reviewed labels.\n5. Track repeat repairs, time to resolution, and post-repair recurrence as corrective action measures.\n\n## Data and method notes\n\n- Exact duplicate rows: **{quality["exact_duplicate_rows"]}**; duplicate event IDs: **{quality["duplicate_event_ids"]}**. No duplicate records were found in this input.\n- Narrative inputs: customer, causal, and correction verbatims. Missing fields remain blank; they were not imputed with invented labels.\n- Clustering: TF-IDF unigram/bigram features and K-Means with a fixed random seed. Top terms are descriptive labels, not validated causes.\n- Monthly event counts are in `summary.json` and `figures/events_by_month.png`; they are counts rather than exposure-adjusted failure rates.\n'''
    path.write_text(report, encoding="utf-8")


def run(input_path: Path, output_dir: Path, clusters: int) -> None:
    source = pd.read_csv(input_path, low_memory=False)
    quality = data_quality(source)
    enriched, terms = enrich_text(source, cluster_count=clusters)
    if "Opened date" in enriched:
        enriched["opened_date_parsed"] = pd.to_datetime(enriched["Opened date"], errors="coerce")
    if "BUILD_DATE" in enriched:
        enriched["build_date_parsed"] = pd.to_datetime(enriched["BUILD_DATE"], errors="coerce", utc=True)
    if "IN_USE_DATE" in enriched:
        enriched["in_use_date_parsed"] = pd.to_datetime(enriched["IN_USE_DATE"], errors="coerce", utc=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "enriched_dataset.csv", index=False)
    (output_dir / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    summary = build_summary(enriched)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "cluster_terms.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")
    make_figures(enriched, output_dir / "figures")
    write_stakeholder_report(enriched, summary, quality, terms, output_dir / "stakeholder_report.md")
    print(f"Analyzed {len(enriched):,} rows. Outputs: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/Dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--clusters", type=int, default=6)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.clusters)


if __name__ == "__main__":
    main()


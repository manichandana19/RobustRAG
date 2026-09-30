from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hallucination_detector.evaluation.eda import (
    load_eda_data,
)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["answer_context_ratio"] = (
        df["answer_words"]
        / df["context_words"].clip(lower=1)
    )

    df["length_difference"] = (
        df["answer_words"] - df["context_words"]
    )

    return df


def model_task_heatmap(
    df: pd.DataFrame,
    output: Path,
) -> None:
    table = (
        df.groupby(["model", "task_type"])["hallucinated"]
        .mean()
        .unstack()
        * 100
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(table.values)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)

    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(
                j,
                i,
                f"{table.iloc[i, j]:.1f}%",
                ha="center",
                va="center",
            )

    ax.set_title(
        "Hallucination Rate: Model × Task Type"
    )
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Generation Model")

    fig.colorbar(
        image,
        ax=ax,
        label="Hallucination Rate (%)",
    )

    save_figure(
        fig,
        output / "11_model_task_heatmap.png",
    )


def length_overlap_scatter(
    df: pd.DataFrame,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for label in ["Supported", "Hallucinated"]:
        subset = df[df["label"] == label]

        ax.scatter(
            subset["answer_words"],
            subset["lexical_overlap"],
            alpha=0.18,
            s=10,
            label=label,
        )

    ax.set_title(
        "Answer Length × Lexical Grounding"
    )
    ax.set_xlabel("Answer Length (words)")
    ax.set_ylabel("Lexical Overlap")
    ax.legend()

    save_figure(
        fig,
        output / "12_answer_length_vs_overlap.png",
    )


def binned_risk_surface(
    df: pd.DataFrame,
    output: Path,
) -> None:
    working = df.copy()

    working["length_bin"] = pd.qcut(
        working["answer_words"],
        q=8,
        duplicates="drop",
    )

    working["overlap_bin"] = pd.qcut(
        working["lexical_overlap"],
        q=8,
        duplicates="drop",
    )

    table = (
        working.groupby(
            ["length_bin", "overlap_bin"],
            observed=True,
        )["hallucinated"]
        .mean()
        .unstack()
        * 100
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    image = ax.imshow(
        table.values,
        aspect="auto",
    )

    ax.set_title(
        "Hallucination Risk Surface: "
        "Answer Length × Lexical Overlap"
    )

    ax.set_xlabel(
        "Lexical Overlap Quantile"
    )
    ax.set_ylabel(
        "Answer Length Quantile"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Hallucination Rate (%)",
    )

    save_figure(
        fig,
        output / "13_length_overlap_risk_surface.png",
    )


def context_answer_surface(
    df: pd.DataFrame,
    output: Path,
) -> None:
    working = df.copy()

    working["context_bin"] = pd.qcut(
        working["context_words"],
        q=8,
        duplicates="drop",
    )

    working["answer_bin"] = pd.qcut(
        working["answer_words"],
        q=8,
        duplicates="drop",
    )

    table = (
        working.groupby(
            ["context_bin", "answer_bin"],
            observed=True,
        )["hallucinated"]
        .mean()
        .unstack()
        * 100
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    image = ax.imshow(
        table.values,
        aspect="auto",
    )

    ax.set_title(
        "Hallucination Risk: Context Length × Answer Length"
    )

    ax.set_xlabel(
        "Answer Length Quantile"
    )
    ax.set_ylabel(
        "Context Length Quantile"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Hallucination Rate (%)",
    )

    save_figure(
        fig,
        output / "14_context_answer_risk_surface.png",
    )


def model_length_interaction(
    df: pd.DataFrame,
    output: Path,
) -> None:
    working = df.copy()

    working["answer_bin"] = pd.qcut(
        working["answer_words"],
        q=6,
        duplicates="drop",
    )

    table = (
        working.groupby(
            ["model", "answer_bin"],
            observed=True,
        )["hallucinated"]
        .mean()
        .unstack()
        * 100
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in table.index:
        ax.plot(
            range(len(table.columns)),
            table.loc[model],
            marker="o",
            label=model,
        )

    ax.set_title(
        "Model × Answer Length Interaction"
    )
    ax.set_xlabel("Answer Length Quantile")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.legend()

    save_figure(
        fig,
        output / "15_model_length_interaction.png",
    )


def model_temperature_interaction(
    df: pd.DataFrame,
    output: Path,
) -> None:
    table = (
        df.groupby(
            ["model", "temperature"]
        )["hallucinated"]
        .mean()
        .unstack()
        * 100
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in table.index:
        ax.plot(
            table.columns,
            table.loc[model],
            marker="o",
            label=model,
        )

    ax.set_title(
        "Model × Temperature Interaction"
    )
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.legend()

    save_figure(
        fig,
        output / "16_model_temperature_interaction.png",
    )


def source_agreement(
    df: pd.DataFrame,
    output: Path,
) -> None:
    source_counts = (
        df.groupby("source_id")["hallucinated"]
        .sum()
    )

    distribution = (
        source_counts
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        distribution.index,
        distribution.values,
    )

    ax.set_title(
        "Cross-Model Hallucination Agreement per Source"
    )
    ax.set_xlabel(
        "Number of Models Hallucinating on the Same Source"
    )
    ax.set_ylabel("Number of Sources")

    save_figure(
        fig,
        output / "17_source_level_agreement.png",
    )


def model_disagreement(
    df: pd.DataFrame,
    output: Path,
) -> None:
    pivot = df.pivot(
        index="source_id",
        columns="model",
        values="hallucinated",
    )

    models = list(pivot.columns)

    disagreement = pd.DataFrame(
        index=models,
        columns=models,
        dtype=float,
    )

    for model_a, model_b in combinations(
        models,
        2,
    ):
        valid = pivot[[model_a, model_b]].dropna()

        rate = (
            valid[model_a]
            != valid[model_b]
        ).mean() * 100

        disagreement.loc[
            model_a,
            model_b,
        ] = rate

        disagreement.loc[
            model_b,
            model_a,
        ] = rate

    # Change this:
    # np.fill_diagonal(
    #     disagreement.values,
    #     0,
    # )

    # To this:
    disagreement_arr = disagreement.to_numpy(copy=True)
    np.fill_diagonal(disagreement_arr, 0)
    disagreement = pd.DataFrame(disagreement_arr, index=disagreement.index, columns=disagreement.columns)

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        disagreement.values,
    )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        models,
        rotation=30,
        ha="right",
    )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(
                j,
                i,
                f"{disagreement.iloc[i, j]:.1f}",
                ha="center",
                va="center",
            )

    ax.set_title(
        "Pairwise Model Disagreement on the Same Sources"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Disagreement (%)",
    )

    save_figure(
        fig,
        output / "18_model_disagreement_matrix.png",
    )


def annotation_model_heatmap(
    df: pd.DataFrame,
    output: Path,
) -> None:
    rows = []

    for _, row in df.iterrows():
        for annotation in row["annotations"]:
            rows.append(
                {
                    "model": row["model"],
                    "label_type": annotation.get(
                        "label_type",
                        "Unknown",
                    ),
                }
            )

    annotations = pd.DataFrame(rows)

    if annotations.empty:
        return

    table = pd.crosstab(
        annotations["model"],
        annotations["label_type"],
        normalize="index",
    ) * 100

    fig, ax = plt.subplots(figsize=(11, 6))

    image = ax.imshow(
        table.values,
        aspect="auto",
    )

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(
        table.columns,
        rotation=30,
        ha="right",
    )

    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(
                j,
                i,
                f"{table.iloc[i, j]:.1f}",
                ha="center",
                va="center",
            )

    ax.set_title(
        "Hallucination Type Composition by Model"
    )
    ax.set_xlabel("Annotation Type")
    ax.set_ylabel("Model")

    fig.colorbar(
        image,
        ax=ax,
        label="Percentage of Model's Annotations",
    )

    save_figure(
        fig,
        output / "19_annotation_type_model_heatmap.png",
    )


def generate_cross_feature_summary(
    df: pd.DataFrame,
    output: Path,
) -> None:
    source_counts = (
        df.groupby("source_id")["hallucinated"]
        .sum()
    )

    model_disagreement = df.pivot(
        index="source_id",
        columns="model",
        values="hallucinated",
    )

    pairwise_rows = []

    for model_a, model_b in combinations(
        model_disagreement.columns,
        2,
    ):
        pair = model_disagreement[
            [model_a, model_b]
        ].dropna()

        pairwise_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "disagreement_rate": (
                    pair[model_a] != pair[model_b]
                ).mean(),
            }
        )

    pairwise = pd.DataFrame(pairwise_rows)

    summary = [
        "# Cross-Feature EDA",
        "",
        "## Derived Features",
        "",
        "- Answer/context length ratio",
        "- Answer/context length difference",
        "",
        "## Source-Level Agreement",
        "",
        f"- Sources with all 6 models supported: "
        f"{(source_counts == 0).sum():,}",
        f"- Sources with all 6 models hallucinating: "
        f"{(source_counts == 6).sum():,}",
        "",
        "## Pairwise Model Disagreement",
        "",
        pairwise.to_markdown(index=False),
        "",
        "## Important Interpretation",
        "",
        "These analyses are descriptive. "
        "Interactions can reveal dataset structure, "
        "but they do not establish causal effects.",
        "Model and temperature are especially likely "
        "to be confounded by the original generation setup.",
    ]

    (
        output / "cross_feature_summary.md"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed",
        type=Path,
        default=Path(
            "data/processed/ragtruth.jsonl"
        ),
    )

    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(
            "data/raw/source_info.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/eda/cross_feature"
        ),
    )

    args = parser.parse_args()

    df = load_eda_data(
        args.processed,
        args.sources,
    )

    df = prepare_features(df)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_task_heatmap(df, args.output)
    length_overlap_scatter(df, args.output)
    binned_risk_surface(df, args.output)
    context_answer_surface(df, args.output)
    model_length_interaction(df, args.output)
    model_temperature_interaction(df, args.output)
    source_agreement(df, args.output)
    model_disagreement(df, args.output)
    annotation_model_heatmap(df, args.output)

    generate_cross_feature_summary(
        df,
        args.output,
    )

    print(
        f"Cross-feature EDA complete: {args.output}"
    )


if __name__ == "__main__":
    main()
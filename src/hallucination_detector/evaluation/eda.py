from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_eda_data(
    processed_path: str | Path,
    source_path: str | Path,
) -> pd.DataFrame:
    """
    Load the finalized processed dataset and join source metadata
    required for exploratory analysis.
    """

    processed_path = Path(processed_path)
    source_path = Path(source_path)

    records = pd.read_json(
        processed_path,
        lines=True,
    )

    sources = pd.read_json(
        source_path,
        lines=True,
    )

    source_metadata = sources[
        [
            "source_id",
            "task_type",
            "source",
        ]
    ].copy()

    source_metadata["source_id"] = (
        source_metadata["source_id"].astype(str)
    )

    records["source_id"] = (
        records["source_id"].astype(str)
    )

    df = records.merge(
        source_metadata,
        on="source_id",
        how="left",
        validate="many_to_one",
    )

    df["label"] = np.where(
        df["hallucinated"],
        "Hallucinated",
        "Supported",
    )

    df["answer_words"] = df["answer"].map(
        lambda x: len(str(x).split())
    )

    df["context_words"] = df["context"].map(
        lambda x: len(str(x).split())
    )

    df["question_words"] = df["question"].map(
        lambda x: len(str(x).split())
    )

    df["annotation_count"] = df["annotations"].map(len)

    df["annotation_density"] = (
        df["annotation_count"]
        / df["answer_words"].clip(lower=1)
    )

    df["lexical_overlap"] = df.apply(
        _calculate_lexical_overlap,
        axis=1,
    )

    return df


def _calculate_lexical_overlap(row: pd.Series) -> float:
    """
    Calculate the proportion of unique answer words that also
    appear in the context.

    This is intentionally a simple lexical grounding signal.
    It is NOT a semantic similarity metric.
    """

    answer_words = {
        word.lower().strip(".,!?;:\"'()[]{}")
        for word in str(row["answer"]).split()
    }

    context_words = {
        word.lower().strip(".,!?;:\"'()[]{}")
        for word in str(row["context"]).split()
    }

    answer_words.discard("")
    context_words.discard("")

    if not answer_words:
        return 0.0

    return len(answer_words & context_words) / len(answer_words)


def _save_figure(
    fig: plt.Figure,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_class_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title("Hallucination Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Responses")

    _save_figure(
        fig,
        output_dir / "01_class_distribution.png",
    )


def plot_split_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = pd.crosstab(
        df["split"],
        df["label"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Class Distribution Across Dataset Splits")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of Responses")
    ax.legend(title="Class")

    _save_figure(
        fig,
        output_dir / "02_split_distribution.png",
    )


def plot_answer_length(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for label in ["Supported", "Hallucinated"]:
        values = df.loc[
            df["label"] == label,
            "answer_words",
        ]

        ax.hist(
            values,
            bins=40,
            alpha=0.55,
            label=label,
        )

    ax.set_title("Answer Length by Hallucination Class")
    ax.set_xlabel("Answer Length (words)")
    ax.set_ylabel("Number of Responses")
    ax.legend()

    _save_figure(
        fig,
        output_dir / "03_answer_length_distribution.png",
    )


def plot_context_length(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        df["context_words"],
        bins=50,
    )

    ax.axvline(
        df["context_words"].median(),
        linestyle="--",
        label=f"Median: {df['context_words'].median():.0f}",
    )

    ax.set_title("Context Length Distribution")
    ax.set_xlabel("Context Length (words)")
    ax.set_ylabel("Number of Responses")
    ax.legend()

    _save_figure(
        fig,
        output_dir / "04_context_length_distribution.png",
    )


def plot_model_hallucination_rate(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    rates = (
        df.groupby("model")["hallucinated"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        rates.index,
        rates.values,
    )

    ax.set_title("Hallucination Rate by Generation Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Hallucination Rate (%)")

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    _save_figure(
        fig,
        output_dir / "05_model_hallucination_rate.png",
    )


def plot_length_binned_rate(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Question:
    Does hallucination frequency change as generated answers become longer?
    """

    working = df.copy()

    working["answer_length_bin"] = pd.qcut(
        working["answer_words"],
        q=10,
        duplicates="drop",
    )

    grouped = (
        working.groupby(
            "answer_length_bin",
            observed=True,
        )["hallucinated"]
        .mean()
        * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        range(len(grouped)),
        grouped.values,
        marker="o",
    )

    ax.set_title(
        "Hallucination Rate Across Answer-Length Deciles"
    )
    ax.set_xlabel("Answer-Length Decile")
    ax.set_ylabel("Hallucination Rate (%)")

    _save_figure(
        fig,
        output_dir / "06_length_binned_hallucination_rate.png",
    )


def plot_lexical_overlap(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Question:
    Do hallucinated answers have weaker lexical grounding in context?
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    data = [
        df.loc[
            df["label"] == label,
            "lexical_overlap",
        ].values
        for label in ["Supported", "Hallucinated"]
    ]

    ax.boxplot(
        data,
        tick_labels=[
            "Supported",
            "Hallucinated",
        ],
    )

    ax.set_title(
        "Lexical Overlap Between Answer and Context"
    )
    ax.set_xlabel("Class")
    ax.set_ylabel(
        "Fraction of Unique Answer Words Found in Context"
    )

    _save_figure(
        fig,
        output_dir / "07_lexical_overlap.png",
    )


def plot_annotation_types(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Question:
    What kinds of hallucination annotations dominate the dataset?
    """

    counter: Counter[str] = Counter()

    for annotations in df["annotations"]:
        for annotation in annotations:
            label_type = annotation.get(
                "label_type",
                "Unknown",
            )

            counter[str(label_type)] += 1

    counts = pd.Series(counter).sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        counts.index,
        counts.values,
    )

    ax.invert_yaxis()

    ax.set_title("Hallucination Annotation Types")
    ax.set_xlabel("Annotation Count")
    ax.set_ylabel("Annotation Type")

    _save_figure(
        fig,
        output_dir / "08_annotation_types.png",
    )


def plot_temperature_rate(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Question:
    Does generation temperature correspond to different
    hallucination rates?
    """

    grouped = (
        df.groupby("temperature")["hallucinated"]
        .mean()
        .sort_index()
        * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        grouped.index,
        grouped.values,
        marker="o",
    )

    ax.set_title(
        "Hallucination Rate by Generation Temperature"
    )
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Hallucination Rate (%)")

    _save_figure(
        fig,
        output_dir / "09_temperature_hallucination_rate.png",
    )


def plot_task_type_rate(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Question:
    Are some RAGTruth task types associated with different
    hallucination rates?
    """

    grouped = (
        df.groupby("task_type")["hallucinated"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    # Ignore extremely tiny groups if the dataset happens to contain
    # them. The threshold is intentionally low enough to preserve
    # useful task types.
    grouped = grouped[grouped["count"] >= 20]

    rates = grouped["mean"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        rates.index,
        rates.values,
    )

    ax.invert_yaxis()

    ax.set_title(
        "Hallucination Rate by RAGTruth Task Type"
    )
    ax.set_xlabel("Hallucination Rate (%)")
    ax.set_ylabel("Task Type")

    _save_figure(
        fig,
        output_dir / "10_task_type_hallucination_rate.png",
    )


def calculate_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce compact numerical summaries that accompany the figures.
    """

    rows = []

    for label in ["Supported", "Hallucinated"]:
        subset = df[df["label"] == label]

        rows.append(
            {
                "group": label,
                "records": len(subset),
                "answer_words_mean": subset[
                    "answer_words"
                ].mean(),
                "answer_words_median": subset[
                    "answer_words"
                ].median(),
                "context_words_mean": subset[
                    "context_words"
                ].mean(),
                "context_words_median": subset[
                    "context_words"
                ].median(),
                "lexical_overlap_mean": subset[
                    "lexical_overlap"
                ].mean(),
                "annotation_density_mean": subset[
                    "annotation_density"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def generate_summary(
    df: pd.DataFrame,
    statistics: pd.DataFrame,
) -> str:
    class_counts = df["label"].value_counts()

    hallucination_rate = (
        df["hallucinated"].mean() * 100
    )

    model_rates = (
        df.groupby("model")["hallucinated"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    task_rates = (
        df.groupby("task_type")["hallucinated"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    annotation_counter: Counter[str] = Counter()

    for annotations in df["annotations"]:
        for annotation in annotations:
            annotation_counter[
                str(
                    annotation.get(
                        "label_type",
                        "Unknown",
                    )
                )
            ] += 1

    lines = [
        "# RAGTruth Exploratory Data Analysis",
        "",
        "## Dataset Overview",
        "",
        f"- Total responses: {len(df):,}",
        f"- Supported: {class_counts.get('Supported', 0):,}",
        f"- Hallucinated: {class_counts.get('Hallucinated', 0):,}",
        f"- Overall hallucination rate: {hallucination_rate:.2f}%",
        f"- Unique sources: {df['source_id'].nunique():,}",
        f"- Generation models: {df['model'].nunique()}",
        f"- Task types: {df['task_type'].nunique()}",
        "",
        "## Questions Explored",
        "",
        "1. Class balance",
        "2. Split composition",
        "3. Answer-length differences",
        "4. Context-length distribution",
        "5. Model-level hallucination rates",
        "6. Hallucination rate across answer-length deciles",
        "7. Lexical grounding differences",
        "8. Hallucination annotation taxonomy",
        "9. Temperature relationship",
        "10. Task-type relationship",
        "",
        "## Model-Level Rates",
        "",
    ]

    for model, rate in model_rates.items():
        lines.append(
            f"- {model}: {rate:.2f}%"
        )

    lines.extend(
        [
            "",
            "## Task-Type Rates",
            "",
        ]
    )

    for task_type, rate in task_rates.items():
        lines.append(
            f"- {task_type}: {rate:.2f}%"
        )

    lines.extend(
        [
            "",
            "## Annotation Types",
            "",
        ]
    )

    for label_type, count in annotation_counter.most_common():
        lines.append(
            f"- {label_type}: {count:,}"
        )

    lines.extend(
        [
            "",
            "## Numerical Summary",
            "",
            statistics.to_markdown(index=False),
            "",
            "## Interpretation Note",
            "",
            "These are descriptive dataset-level observations. "
            "They do not establish causal relationships. "
            "In particular, model, temperature, task type, "
            "answer length, and hallucination behavior can be "
            "confounded by how the original dataset was generated.",
        ]
    )

    return "\n".join(lines)


def run_eda(
    processed_path: str | Path,
    source_path: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_eda_data(
        processed_path,
        source_path,
    )

    plot_class_distribution(df, figure_dir)
    plot_split_distribution(df, figure_dir)
    plot_answer_length(df, figure_dir)
    plot_context_length(df, figure_dir)
    plot_model_hallucination_rate(df, figure_dir)
    plot_length_binned_rate(df, figure_dir)
    plot_lexical_overlap(df, figure_dir)
    plot_annotation_types(df, figure_dir)
    plot_temperature_rate(df, figure_dir)
    plot_task_type_rate(df, figure_dir)

    statistics = calculate_statistics(df)

    statistics.to_csv(
        output_dir / "eda_statistics.csv",
        index=False,
    )

    summary = generate_summary(
        df,
        statistics,
    )

    (output_dir / "eda_summary.md").write_text(
        summary,
        encoding="utf-8",
    )

    print(f"Records analysed: {len(df):,}")
    print(f"Figures written to: {figure_dir}")
    print(
        f"Summary written to: "
        f"{output_dir / 'eda_summary.md'}"
    )
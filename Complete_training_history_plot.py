import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_data(json_path):
    """Load training summary from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_phase_metrics(phase_data):
    """Extract metrics from phase results"""
    results_dict = phase_data["results_dict"]
    return {
        "precision": results_dict["metrics/precision(B)"],
        "recall": results_dict["metrics/recall(B)"],
        "mAP50": results_dict["metrics/mAP50(B)"],
        "mAP50-95": results_dict["metrics/mAP50-95(B)"],
        "fitness": results_dict["fitness"],
    }


def plot_metrics_over_phases(training_data, output_dir):
    """Plot main metrics evolution across training phases"""
    phases = training_data["phase_results"]
    phase_numbers = [p["phase"] for p in phases]

    metrics = {"precision": [], "recall": [], "mAP50": [], "mAP50-95": [], "fitness": []}

    for phase in phases:
        phase_metrics = extract_phase_metrics(phase)
        for key in metrics.keys():
            metrics[key].append(phase_metrics[key])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Training Metrics Evolution Across Phases", fontsize=16, fontweight="bold")

    plot_configs = [
        ("precision", "Precision", "blue"),
        ("recall", "Recall", "green"),
        ("mAP50", "mAP@0.5", "red"),
        ("mAP50-95", "mAP@0.5:0.95", "purple"),
        ("fitness", "Fitness Score", "orange"),
    ]

    for idx, (metric_key, title, color) in enumerate(plot_configs):
        ax = axes[idx // 3, idx % 3]
        ax.plot(
            phase_numbers,
            metrics[metric_key],
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
            label=title,
        )
        ax.set_xlabel("Training Phase", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        for i, val in enumerate(metrics[metric_key]):
            ax.annotate(
                f"{val:.3f}",
                xy=(phase_numbers[i], val),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_map(training_data, output_dir):
    """Plot mAP per class for each phase"""
    phases = training_data["phase_results"]
    class_names = training_data["dataset_info"]["classes"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("mAP@0.5 per Class Across Training Phases", fontsize=16, fontweight="bold")

    for idx, phase in enumerate(phases):
        ax = axes[idx // 3, idx % 3]
        maps = phase["maps"]

        bars = ax.bar(range(len(class_names)), maps, color="steelblue", alpha=0.7)
        ax.set_xlabel("Class", fontsize=10)
        ax.set_ylabel("mAP@0.5", fontsize=10)
        ax.set_title(f'Phase {phase["phase"]}', fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, maps):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "per_class_map.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_comparison(training_data, output_dir):
    """Compare class performance between first and last phase"""
    phases = training_data["phase_results"]
    class_names = training_data["dataset_info"]["classes"]

    first_phase = phases[0]["maps"]
    last_phase = phases[-1]["maps"]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(
        x - width / 2, first_phase, width, label="Phase 1", color="lightcoral", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2,
        last_phase,
        width,
        label=f'Phase {phases[-1]["phase"]}',
        color="lightgreen",
        alpha=0.8,
    )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("Class Performance: First vs Last Phase", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "class_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_configuration(training_data, output_dir):
    """Plot training configuration and dataset information"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Progressive unfreezing schedule
    schedule = training_data["progressive_unfreezing"]["schedule"]
    epochs = [int(k) for k in schedule.keys()]
    unfreezing = [schedule[k] for k in schedule.keys()]

    ax1.plot(epochs, unfreezing, marker="o", linewidth=2, markersize=8, color="darkblue")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Unfreezing Ratio", fontsize=12)
    ax1.set_title("Progressive Unfreezing Schedule", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    for i, (e, u) in enumerate(zip(epochs, unfreezing)):
        ax1.annotate(
            f"{u}", xy=(e, u), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9
        )

    # Class distribution
    class_counts = training_data["phase_results"][0]["nt_per_class"]
    class_names = training_data["dataset_info"]["classes"]

    bars = ax2.barh(class_names, class_counts, color="teal", alpha=0.7)
    ax2.set_xlabel("Number of Instances", fontsize=12)
    ax2.set_ylabel("Class", fontsize=12)
    ax2.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    for bar, count in zip(bars, class_counts):
        width = bar.get_width()
        ax2.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count}",
            ha="left",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_dir / "training_configuration.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_summary_table(training_data, output_dir):
    """Create a summary table with all metrics"""
    phases = training_data["phase_results"]

    fig, ax = plt.subplots(figsize=(12, len(phases) * 0.8 + 2))
    ax.axis("tight")
    ax.axis("off")

    headers = ["Phase", "Epochs", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Fitness"]
    table_data = []

    for phase in phases:
        metrics = extract_phase_metrics(phase)
        row = [
            f"Phase {phase['phase']}",
            f"{phase['start_epoch']}-{phase['start_epoch'] + phase['epochs']}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['mAP50']:.4f}",
            f"{metrics['mAP50-95']:.4f}",
            f"{metrics['fitness']:.4f}",
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["lightblue"] * len(headers),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(headers)):
        table[(0, i)].set_facecolor("steelblue")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Training Summary Table", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "metrics_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_weights(training_data, output_dir):
    """Plot class weights used for training"""
    class_weights = training_data["dataset_info"]["class_weights"]
    class_names = training_data["dataset_info"]["classes"]

    weights = [class_weights[str(i)] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(class_names, weights, color="darkorange", alpha=0.7)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.set_title("Class Weights for Imbalanced Dataset", fontsize=14, fontweight="bold")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{weight:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "class_weights.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    json_path = Path("Models/YOLOv12_Detection_11-06-2025_04-13-31/training_summary.json")
    output_dir = Path("Complete_training_plots")
    output_dir.mkdir(exist_ok=True)

    print("Loading training data...")
    training_data = load_training_data(json_path)

    print("Generating plots...")

    print("  - Metrics evolution...")
    plot_metrics_over_phases(training_data, output_dir)

    print("  - Per-class mAP...")
    plot_per_class_map(training_data, output_dir)

    print("  - Class comparison...")
    plot_class_comparison(training_data, output_dir)

    print("  - Training configuration...")
    plot_training_configuration(training_data, output_dir)

    print("  - Metrics summary table...")
    plot_metrics_summary_table(training_data, output_dir)

    print("  - Class weights...")
    plot_class_weights(training_data, output_dir)

    print(f"\nAll plots saved successfully in '{output_dir}' directory")
    print(f"Total plots generated: 6")


if __name__ == "__main__":
    main()

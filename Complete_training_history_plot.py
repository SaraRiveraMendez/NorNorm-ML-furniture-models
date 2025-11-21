import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_data(json_path):
    """Load training summary from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_phase_metrics(phase_data):
    """Extract metrics from phase results by parsing the results string"""
    results_str = phase_data.get("results", "")

    if not results_str:
        print(f"Warning: No results found in phase {phase_data.get('phase', 'unknown')}")
        return None

    try:
        # Try to extract from string format first
        metrics = {}

        # Pattern to find results_dict section
        results_dict_pattern = r"results_dict: \{([^}]+)\}"
        match = re.search(results_dict_pattern, results_str)

        if match:
            results_content = match.group(1)

            # Extract individual metrics
            precision_match = re.search(r"'metrics/precision\(B\)': ([\d.]+)", results_content)
            recall_match = re.search(r"'metrics/recall\(B\)': ([\d.]+)", results_content)
            map50_match = re.search(r"'metrics/mAP50\(B\)': ([\d.]+)", results_content)
            map50_95_match = re.search(r"'metrics/mAP50-95\(B\)': ([\d.]+)", results_content)
            fitness_match = re.search(r"'fitness': ([\d.]+)", results_content)

            if all([precision_match, recall_match, map50_match, map50_95_match, fitness_match]):
                return {
                    "precision": float(precision_match.group(1)),
                    "recall": float(recall_match.group(1)),
                    "mAP50": float(map50_match.group(1)),
                    "mAP50-95": float(map50_95_match.group(1)),
                    "fitness": float(fitness_match.group(1)),
                }

        # Alternative: check if results is already a dict
        if isinstance(phase_data.get("results"), dict):
            results = phase_data["results"]
            return {
                "precision": results.get("metrics/precision(B)", 0),
                "recall": results.get("metrics/recall(B)", 0),
                "mAP50": results.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.get("metrics/mAP50-95(B)", 0),
                "fitness": results.get("fitness", 0),
            }

        print(f"Warning: Could not parse metrics from phase {phase_data.get('phase', 'unknown')}")
        return None

    except Exception as e:
        print(f"Error parsing metrics from phase {phase_data.get('phase', 'unknown')}: {e}")
        return None


def plot_metrics_over_phases(training_data, output_dir):
    """Plot main metrics evolution across training phases"""
    phases = training_data["phase_results"]

    # Filter phases that have valid metrics
    valid_phases = []
    valid_metrics = {"precision": [], "recall": [], "mAP50": [], "mAP50-95": [], "fitness": []}

    for phase in phases:
        phase_metrics = extract_phase_metrics(phase)
        if phase_metrics:
            valid_phases.append(phase)
            for key in valid_metrics.keys():
                valid_metrics[key].append(phase_metrics[key])

    if not valid_phases:
        print("No valid phases with metrics found!")
        return

    phase_numbers = [p["phase"] for p in valid_phases]

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
            valid_metrics[metric_key],
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

        for i, val in enumerate(valid_metrics[metric_key]):
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


def plot_metrics_summary_table(training_data, output_dir):
    """Create a summary table with all metrics"""
    phases = training_data["phase_results"]

    # Filter phases with valid metrics
    valid_phases = []
    for phase in phases:
        if extract_phase_metrics(phase):
            valid_phases.append(phase)

    if not valid_phases:
        print("No valid phases for summary table")
        return

    fig, ax = plt.subplots(figsize=(12, len(valid_phases) * 0.8 + 2))
    ax.axis("tight")
    ax.axis("off")

    headers = ["Phase", "Epochs", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Fitness"]
    table_data = []

    for phase in valid_phases:
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


def safe_plot_function(plot_func, training_data, output_dir, plot_name):
    """Wrapper para manejar errores en las funciones de plotting"""
    try:
        plot_func(training_data, output_dir)
        print(f"  - {plot_name}... ✓")
        return True
    except Exception as e:
        print(f"  - {plot_name}... ✗ (Error: {e})")
        return False


def debug_data_structure(training_data):
    """Debug function to understand the data structure"""
    print("\n=== DEBUG DATA STRUCTURE ===")
    phases = training_data["phase_results"]

    print(f"Number of phases: {len(phases)}")

    for i, phase in enumerate(phases):
        print(f"\n--- Phase {i} ---")
        print(f"Keys: {list(phase.keys())}")

        if "results" in phase:
            results = phase["results"]
            print(f"Results type: {type(results)}")
            if isinstance(results, str):
                print("Results is a string (needs parsing)")
                # Print first 200 chars
                print(f"Results preview: {results[:200]}...")
            elif isinstance(results, dict):
                print("Results is a dict")
                print(f"Results keys: {list(results.keys())}")

        if "maps" in phase:
            maps = phase["maps"]
            print(f"Maps type: {type(maps)}")
            print(f"Maps length: {len(maps) if hasattr(maps, '__len__') else 'N/A'}")
            if hasattr(maps, "__len__") and len(maps) > 0:
                print(f"Maps preview: {maps[:5]}...")  # First 5 values


def main():
    json_path = Path(  # Change the plot
        "C:/Users/rsara/OneDrive/Documents/NorNorm/NorNorm-ML-furniture-models/Models/YOLOv12_Detection_11-11-2025_03-40-12/training_summary.json"
    )
    output_dir = Path("Complete_training_plots")
    output_dir.mkdir(exist_ok=True)

    print("Loading training data...")

    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return

    try:
        training_data = load_training_data(json_path)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Debug: check data structure
    debug_data_structure(training_data)

    print("\nGenerating plots...")

    plot_functions = [
        (plot_metrics_over_phases, "Metrics evolution"),
        (plot_metrics_summary_table, "Metrics summary table"),
        (plot_class_weights, "Class weights"),
    ]

    successful_plots = 0
    for plot_func, plot_name in plot_functions:
        if safe_plot_function(plot_func, training_data, output_dir, plot_name):
            successful_plots += 1

    print(f"\nPlots generation completed!")
    print(f"Successful: {successful_plots}/{len(plot_functions)}")
    print(f"Output directory: '{output_dir.absolute()}'")


if __name__ == "__main__":
    main()

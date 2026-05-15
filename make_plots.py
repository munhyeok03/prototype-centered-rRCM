import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _placeholder(path, title, message):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _line_plot(df, y, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in df.groupby("method"):
        group = group.sort_values("sigma")
        ax.plot(group["sigma"], group[y], marker="o", label=method)
    ax.set_title(title)
    ax.set_xlabel("sigma")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        clean = metrics[metrics["metric"] == "clean_accuracy"]
        if len(clean):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(clean["method"], clean["value"])
            ax.set_ylabel("accuracy (%)")
            ax.set_title("Clean Accuracy")
            ax.tick_params(axis="x", labelrotation=20)
            fig.tight_layout()
            fig.savefig(plots_dir / "clean_accuracy.png", dpi=160)
            plt.close(fig)
        else:
            _placeholder(plots_dir / "clean_accuracy.png", "Clean Accuracy", "No clean accuracy rows available.")

        noisy = metrics[metrics["metric"].str.startswith("accuracy_noise_sigma_")]
        if len(noisy):
            fig, ax = plt.subplots(figsize=(6, 4))
            for method, group in noisy.groupby("method"):
                group = group.sort_values("sigma")
                ax.plot(group["sigma"], group["value"], marker="o", label=method)
            ax.set_xlabel("sigma")
            ax.set_ylabel("accuracy (%)")
            ax.set_title("Noisy Accuracy vs Sigma")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / "noisy_accuracy_vs_sigma.png", dpi=160)
            plt.close(fig)
        else:
            _placeholder(plots_dir / "noisy_accuracy_vs_sigma.png", "Noisy Accuracy", "No noisy accuracy rows available.")
    else:
        _placeholder(plots_dir / "clean_accuracy.png", "Clean Accuracy", "metrics_summary.csv was not generated.")
        _placeholder(plots_dir / "noisy_accuracy_vs_sigma.png", "Noisy Accuracy", "metrics_summary.csv was not generated.")

    certified_path = results_dir / "certified_summary.csv"
    if certified_path.exists():
        certified = pd.read_csv(certified_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, group in certified.groupby("method"):
            group = group.sort_values("radius")
            ax.plot(group["radius"], group["certified_accuracy"], marker="o", label=method)
        ax.set_xlabel("radius")
        ax.set_ylabel("certified accuracy (%)")
        ax.set_title("Certified Accuracy vs Radius")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "certified_accuracy_vs_radius.png", dpi=160)
        plt.close(fig)
    else:
        _placeholder(
            plots_dir / "certified_accuracy_vs_radius.png",
            "Certified Accuracy vs Radius",
            "Original certification was not run; no certified accuracy is claimed.",
        )

    geometry_path = results_dir / "geometry_summary.csv"
    if geometry_path.exists():
        geometry = pd.read_csv(geometry_path)
        _line_plot(geometry, "prototype_margin", "Prototype Margin vs Sigma", "mean sim_correct - sim_wrong_max", plots_dir / "prototype_margin_vs_sigma.png")
        _line_plot(geometry, "correct_proto_similarity", "Correct Prototype Similarity vs Sigma", "mean cosine similarity", plots_dir / "correct_proto_similarity_vs_sigma.png")
        _line_plot(geometry, "max_wrong_proto_similarity", "Wrong Prototype Similarity vs Sigma", "mean max wrong cosine similarity", plots_dir / "wrong_proto_similarity_vs_sigma.png")
    else:
        _placeholder(plots_dir / "prototype_margin_vs_sigma.png", "Prototype Margin", "geometry_summary.csv was not generated.")
        _placeholder(plots_dir / "correct_proto_similarity_vs_sigma.png", "Correct Prototype Similarity", "geometry_summary.csv was not generated.")
        _placeholder(plots_dir / "wrong_proto_similarity_vs_sigma.png", "Wrong Prototype Similarity", "geometry_summary.csv was not generated.")

    inference_path = results_dir / "inference_cost.csv"
    if inference_path.exists():
        inference = pd.read_csv(inference_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(inference["method"], inference["latency_ms_per_image"])
        ax.set_ylabel("ms / image")
        ax.set_title("One-Shot Inference Latency")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / "inference_latency.png", dpi=160)
        plt.close(fig)
    else:
        _placeholder(plots_dir / "inference_latency.png", "Inference Latency", "inference_cost.csv was not generated.")

    print(f"wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()

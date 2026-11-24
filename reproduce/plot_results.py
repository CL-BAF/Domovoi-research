"""Regenerate publication SVGs from archived JSONL and evaluation results."""

from datetime import datetime
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from publication_data import ARMS, ROOT, load_campaign


FRAMES = ROOT / "experiments/figures"
COLORS = {"B02": "#176993", "B08": "#913e21", "R02x4": "#276d5b"}
LABELS = {"B02": "B02", "B08": "B08", "R02x4": "R02×4"}


def save(fig, name):
    FRAMES.mkdir(parents=True, exist_ok=True)
    stream = StringIO()
    fig.savefig(stream, format="svg", metadata={"Date": None}, facecolor="white")
    svg = "\n".join(line.rstrip() for line in stream.getvalue().splitlines()) + "\n"
    (FRAMES / name).write_text(svg, encoding="utf-8", newline="\n")
    plt.close(fig)


def trace(ax, campaign, key, metric, xlabel="Optimizer update"):
    for name in ARMS:
        run = campaign[name]
        ax.plot([p["step"] for p in run[key]], [p[metric] for p in run[key]],
                color=COLORS[name], label=LABELS[name], alpha=0.92, linewidth=1.5,
                marker="o" if key == "val" else None, markersize=2.7)
    ax.set(xlabel=xlabel, ylabel="Nats / target")


def rate_memory(ax, campaign):
    for name in ARMS:
        run = campaign[name]
        vram = run["summary"]["vram"]
        x = vram["peak_allocated_mb"]
        y = run["step_rate"] / 1000
        ax.scatter(x, y, color=COLORS[name], s=42, zorder=3)
        ax.annotate(LABELS[name], (x, y), xytext=(4, 7), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Peak allocated training memory (MiB)", ylabel="Median update throughput (k tok/s)")
    ax.margins(x=0.14, y=0.16)


def calibration():
    campaign = load_campaign("calibration-v2")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    trace(axes[0], campaign, "train", "train_loss")
    axes[0].set_title("Training loss windows")
    trace(axes[1], campaign, "val", "val_loss")
    axes[1].set_title("Sampled validation")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    rate_memory(axes[2], campaign)
    axes[2].set_title("Training resources")
    fig.suptitle("TinyStories calibration · 200 updates · RTX 5080", fontsize=14, fontweight="bold")
    save(fig, "tinystories-calibration.svg")


def extended():
    campaign = load_campaign("extended-v1")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), layout="constrained")
    trace(axes[0, 0], campaign, "train", "train_loss")
    axes[0, 0].set_title("Training loss windows")
    trace(axes[0, 1], campaign, "val", "val_loss")
    axes[0, 1].set_title("Sampled validation")
    for name in ARMS:
        run = campaign[name]
        steps = sorted(run["eval"])
        axes[1, 0].plot([s * run["tokens_per_update"] / 1e6 for s in steps],
                        [run["eval"][s]["nll"] for s in steps],
                        marker="o", markersize=4, color=COLORS[name], label=LABELS[name])
    axes[1, 0].set(title="Full validation at matched budgets",
                   xlabel="Training target tokens (million)", ylabel="Nats / target")
    axes[1, 0].legend(frameon=False, fontsize=8)
    rate_memory(axes[1, 1], campaign)
    axes[1, 1].set_title("Training resources")
    fig.suptitle("Extended TinyStories · 100M tokens per arm · seed 1234",
                 fontsize=14, fontweight="bold")
    save(fig, "tinystories-extended.svg")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout="constrained")
    for name in ARMS:
        samples = campaign[name]["gpu"]
        times = [datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for row in samples]
        age = [(now-times[0]).total_seconds() for now in times]
        for ax, key in zip(axes, ("temperature.gpu", "utilization.gpu", "memory.used")):
            ax.plot(age, [float(row[key]) for row in samples], linewidth=1.3,
                    color=COLORS[name], label=LABELS[name])
    for ax, title, ylabel in zip(axes,
                                  ("Temperature", "Utilization", "Device memory in use"),
                                  ("°C", "%", "Device-wide used (MiB)")):
        ax.set(title=title, xlabel="Seconds since first sample of each arm", ylabel=ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Extended campaign · NVIDIA device telemetry (sampled)",
                 fontsize=14, fontweight="bold")
    save(fig, "tinystories-extended-telemetry.svg")


if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "domovoi-publication-v1", "font.family": "DejaVu Sans",
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.grid": True, "grid.alpha": .18})
    calibration()
    extended()

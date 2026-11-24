import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recorded metrics")
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--serving", type=Path, nargs="*", default=[])
    parser.add_argument("--reward-log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/plots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluations = [read(path) for path in args.eval]
    first = evaluations[0]
    if any(row["dataset_sha256"] != first["dataset_sha256"] for row in evaluations):
        parser.error("Quality plots require the same dataset")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0f172a",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.prop_cycle": cycler(color=["#3b5680", "#087b75", "#6949a0", "#9a4e24"]),
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "grid.color": "#dbe4ef",
        }
    )
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    tasks = list(first["summary"]["by_task"])
    width = 0.8 / len(evaluations)
    for i, row in enumerate(evaluations):
        scores = [row["summary"]["by_task"][task]["field_macro_f1"] for task in tasks]
        axis.bar(
            [j - 0.4 + (i + 0.5) * width for j in range(len(tasks))],
            [float("nan") if value is None else value for value in scores],
            width=width,
            label=row["model"],
        )
    axis.set(
        xticks=list(range(len(tasks))),
        xticklabels=[task.capitalize() for task in tasks],
        ylim=(0, 1),
        ylabel="Field macro-F1",
    )
    axis.set_title("Extraction quality by task", loc="left", pad=14)
    axis.set_axisbelow(True)
    axis.grid(axis="y")
    axis.tick_params(axis="both", length=0, pad=8)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize="small"
    )
    figure.savefig(args.output / "quality.png", dpi=160)
    plt.close(figure)
    if args.reward_log:
        rewards = defaultdict(list)
        for line in args.reward_log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["step"] is not None:
                rewards[row["step"]].append(row["total"])
        steps = sorted(rewards)
        figure, axis = plt.subplots(figsize=(8, 4.2), layout="constrained")
        axis.plot(steps, [mean(rewards[step]) for step in steps], marker=".")
        axis.set(
            xlabel="GRPO optimizer step",
            ylabel="Mean verifiable reward",
            title="GRPO reward by step",
            ylim=(0, 1),
        )
        figure.savefig(args.output / "reward.png", dpi=160)
        plt.close(figure)
    if args.serving:
        figure, axis = plt.subplots(figsize=(8, 4.2), layout="constrained")
        measured = []
        for path in args.serving:
            serving = read(path)
            candidates = [
                row
                for row in evaluations
                if all(
                    row[key] == serving[key]
                    for key in (
                        "model",
                        "backend",
                        "dataset_sha256",
                        "constrained",
                        "max_tokens",
                        "seed",
                    )
                )
            ]
            cost = serving["summary"].get("gpu_electricity_usd_per_million_requests")
            if not candidates or cost is None:
                continue
            quality = candidates[0]["summary"]["field_macro_f1"]
            if quality is None:
                continue
            measured.append((cost, quality, f"{serving['model']} · c{serving['concurrency']}"))
        for cost, quality, label in measured:
            axis.scatter(cost, quality, label=label)
        frontier = []
        for cost, quality, _ in sorted(measured, key=lambda point: (point[0], -point[1])):
            if not frontier or quality > frontier[-1][1]:
                frontier.append((cost, quality))
        if len(frontier) > 1:
            axis.plot(*zip(*frontier, strict=True), linestyle="--", color="gray")
        if measured:
            axis.set(
                xlabel="GPU electricity USD / million requests",
                ylabel="Field macro-F1",
                title="Serving cost and extraction quality",
            )
            axis.legend(fontsize="small")
            figure.savefig(args.output / "cost-vs-f1.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()

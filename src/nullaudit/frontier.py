"""Run equal protected-byte caps and report a memory/stability/plasticity frontier."""

import argparse
import csv
import json
from pathlib import Path

import torch

from . import cli, core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--methods", choices=core.METHODS, nargs="+", default=["nullaudit", "prior_rsj", "ogd", "gpm", "replay"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for budget in args.budgets:
        for method in args.methods:
            directory = output / str(budget) / method
            cfg = {**base, "memory_bytes": budget}
            try:
                cli.train(cfg, args.data, directory, method)
            except ValueError as error:
                if "budget" not in str(error).lower() and "cap" not in str(error).lower():
                    raise
                rows.append(dict(method=method, byte_cap=budget, status="infeasible", reason=str(error)))
                continue
            metrics = cli.evaluate(directory / "model.pt", Path(args.data) / "test.npz", directory / "evaluation")
            rows.append(dict(method=method, byte_cap=budget, status="computed", **metrics))
    (output / "frontier.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    columns = ["method", "byte_cap", "status", "protected_bytes", "average_task_accuracy", "forgetting", "new_task_accuracy", "gradient_retention", "process_peak_rss_bytes_sampled", "reason"]
    with (output / "frontier.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for method in args.methods:
        selected = [r for r in rows if r["method"] == method and r["status"] == "computed"]
        for axis, metric in zip(axes, ("average_task_accuracy", "forgetting", "gradient_retention")):
            axis.plot([r["protected_bytes"] for r in selected], [r[metric] for r in selected], marker="o", label=method)
            axis.set(xlabel="Retained protection state (bytes)", ylabel=metric)
    axes[-1].legend(fontsize=7)
    if base.get("functional_inputs_only", False):
        fig.suptitle("Generated functional inputs; not paper performance")
    fig.tight_layout()
    fig.savefig(output / "frontier.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"configurations": len(rows), "computed": sum(r["status"] == "computed" for r in rows)}))


if __name__ == "__main__":
    main()

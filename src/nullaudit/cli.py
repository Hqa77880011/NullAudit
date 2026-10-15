import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from . import core


class Arrays(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as source:
            self.data = {
                k: np.asarray(source[k]) for k in source.files if k not in ("group", "split")
            }
        if "x" not in self.data or "y" not in self.data:
            raise ValueError("Dataset requires x and y arrays")
        self.length = len(self.data["x"])
        if any(len(v) != self.length for v in self.data.values()):
            raise ValueError("Every array must share the sample axis")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {k: torch.from_numpy(np.asarray(v[index])) for k, v in self.data.items()}


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def prepare(source, output, seed):
    with np.load(source, allow_pickle=False) as file:
        values = {k: np.asarray(file[k]) for k in file.files}
    n = len(values["x"])
    groups = values.get("group", np.arange(n))
    if "split" in values:
        split = values["split"]
        for group in np.unique(groups):
            if len(np.unique(split[groups == group])) != 1:
                raise ValueError("One source group occurs in multiple splits")
    else:
        unique = np.unique(groups)
        np.random.default_rng(seed).shuffle(unique)
        if len(unique) < 3:
            raise ValueError("At least three independent source groups required")
        cut1, cut2 = max(1, int(0.6 * len(unique))), min(len(unique) - 1, int(0.8 * len(unique)))
        split = np.full(n, 2)
        split[np.isin(groups, unique[:cut1])] = 0
        split[np.isin(groups, unique[cut1:cut2])] = 1
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(("train", "validation", "test")):
        mask = split == index
        if not mask.any():
            raise ValueError("Empty dataset split: " + name)
        np.savez_compressed(
            output / (name + ".npz"),
            **{k: v[mask] for k, v in values.items() if k not in ("split",) and len(v) == n},
        )


def train(config, data, output, method):
    cfg = dict(config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    model = core.Model(cfg, method).to(device)
    train_set = Arrays(Path(data) / "train.npz")
    validation_set = Arrays(Path(data) / "validation.npz")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory: " + str(output))
    if hasattr(core, "fit"):
        return core.fit(model, train_set, validation_set, cfg, output)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, generator=generator)
    validation = DataLoader(validation_set, batch_size=cfg["batch_size"])
    if cfg.get("balance_classes", False):
        counts = np.bincount(train_set.data["y"].astype(int), minlength=cfg["classes"])
        if np.any(counts == 0):
            raise ValueError("Every configured class needs training examples")
        weights = 1 / counts
        cfg["class_weights"] = (weights / weights.mean()).tolist()
    parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("adversary.")
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    adversary = (
        torch.optim.AdamW(model.adversary.parameters(), lr=cfg["learning_rate"])
        if hasattr(model, "adversary")
        else None
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        samples = 0
        start = time.perf_counter()
        for batch in loader:
            batch = to_device(batch, device)
            if adversary is not None:
                adversary.zero_grad()
                adversary_loss = core.adversary_loss(model, batch, cfg)
                adversary_loss.backward()
                adversary.step()
            optimizer.zero_grad()
            loss = core.loss(model, batch, cfg)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch["x"])
            samples += len(batch["x"])
        model.eval()
        valid = 0.0
        count = 0
        with torch.set_grad_enabled(getattr(core, "EVALUATION_NEEDS_GRAD", False)):
            for batch in validation:
                batch = to_device(batch, device)
                value = core.loss(model, batch, cfg)
                valid += float(value.detach()) * len(batch["x"])
                count += len(batch["x"])
        valid /= count
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": total / samples,
            "validation_loss": valid,
            "epoch_seconds": time.perf_counter() - start,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if valid < best:
            best, stale = valid, 0
            torch.save(
                {"state": model.state_dict(), "config": cfg, "method": method}, output / "model.pt"
            )
        else:
            stale += 1
        if stale >= cfg.get("patience", 15):
            break
    if hasattr(core, "calibrate"):
        payload = torch.load(output / "model.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["state"])
        payload["calibration"] = core.calibrate(model, validation_set, cfg)
        torch.save(payload, output / "model.pt")


def evaluate(checkpoint, data, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    cfg["_checkpoint_directory"] = str(Path(checkpoint).parent)
    device = torch.device(cfg["device"])
    model = core.Model(cfg, payload["method"]).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    source = Arrays(data)
    batches = DataLoader(source, batch_size=cfg["batch_size"])
    prediction, elapsed = {}, []
    operations = {"macs": 0}
    hooks = []

    def count_operations(module, inputs, result):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            operations["macs"] += (
                result.numel()
                * (module.in_channels // module.groups)
                * int(np.prod(module.kernel_size))
            )
        elif isinstance(module, torch.nn.Linear):
            operations["macs"] += result.numel() * module.in_features
        elif isinstance(module, torch.nn.GRU):
            value = inputs[0]
            batch, steps = (
                value.shape[:2] if module.batch_first else (value.shape[1], value.shape[0])
            )
            directions = 2 if module.bidirectional else 1
            for layer in range(module.num_layers):
                width = module.input_size if layer == 0 else module.hidden_size * directions
                operations["macs"] += (
                    batch
                    * steps
                    * directions
                    * 3
                    * module.hidden_size
                    * (width + module.hidden_size)
                )

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.GRU)):
            hooks.append(module.register_forward_hook(count_operations))
    total_macs = 0
    try:
        for index, batch in enumerate(batches):
            batch = to_device(batch, device)
            with torch.set_grad_enabled(getattr(core, "EVALUATION_NEEDS_GRAD", False)):
                result = core.predict(model, batch, cfg, payload.get("calibration"))
            with torch.set_grad_enabled(getattr(core, "INFERENCE_NEEDS_GRAD", False)):
                if index == 0:
                    core.inference(model, batch, cfg)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                operations["macs"] = 0
                start = time.perf_counter()
                core.inference(model, batch, cfg)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed.append((time.perf_counter() - start, len(batch["x"])))
                total_macs += operations["macs"]
            for key, value in result.items():
                prediction.setdefault(key, []).append(value.detach().cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    arrays = {key: np.concatenate(values) for key, values in prediction.items()}
    metrics = core.metrics(arrays, source.data, cfg)
    total_time = sum(t for t, _ in elapsed)
    examples = sum(n for _, n in elapsed)
    metrics["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    metrics["inference_seconds_per_example"] = total_time / examples
    metrics["batch_latency_p50_seconds"] = float(np.quantile([t for t, _ in elapsed], 0.5))
    metrics["batch_latency_p95_seconds"] = float(np.quantile([t for t, _ in elapsed], 0.95))
    metrics["inference_examples_per_second"] = examples / total_time
    metrics["conv_linear_gru_macs_per_example"] = total_macs / examples
    metrics["conv_linear_gru_flops_per_example"] = 2 * total_macs / examples
    metrics["process_rss_mib"] = psutil.Process().memory_info().rss / 2**20
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", **arrays)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def plot(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        source = Path(path)
        if (source / "history.jsonl").exists():
            rows = [
                json.loads(x)
                for x in (source / "history.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            axis = [
                r.get("epoch", r.get("round", r.get("task", i + 1))) for i, r in enumerate(rows)
            ]
            for field in (
                "train_loss",
                "validation_loss",
                "validation_accuracy",
                "drift",
                "local_steps",
                "gradient_retention",
                "mean_rank",
            ):
                values = [row.get(field) for row in rows]
                if not any(value is not None for value in values):
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(axis, values)
                ax.set_xlabel("Training progression")
                ax.set_ylabel(field)
                fig.tight_layout()
                fig.savefig(output / (source.name + "_" + field + ".png"), dpi=180)
                plt.close(fig)
    reports = [
        (Path(p).name, json.loads((Path(p) / "metrics.json").read_text(encoding="utf-8")))
        for p in inputs
        if (Path(p) / "metrics.json").exists()
    ]
    keys = sorted(
        {k for _, report in reports for k, v in report.items() if isinstance(v, (int, float))}
    )
    for key in keys:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            [name for name, _ in reports],
            [report.get(key, np.nan) for _, report in reports],
            color="#527c91",
        )
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output / (key + ".png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--source", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--seed", type=int, default=17)
    for name in ("train", "suite"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--data", required=True)
        item.add_argument("--output", required=True)
        item.add_argument(
            "--methods",
            nargs="+",
            choices=core.METHODS,
            default=[core.METHODS[0]] if name == "train" else core.METHODS,
        )
    item = sub.add_parser("evaluate")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--data", required=True)
    item.add_argument("--output", required=True)
    item = sub.add_parser("plot")
    item.add_argument("--inputs", nargs="+", required=True)
    item.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output, args.seed)
    elif args.command in ("train", "suite"):
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for method in args.methods:
            train(cfg, args.data, Path(args.output) / method, method)
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.data, args.output)
    else:
        plot(args.inputs, args.output)


if __name__ == "__main__":
    main()

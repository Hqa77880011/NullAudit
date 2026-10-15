import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vision import classification_metrics


METHODS = ["nullaudit", "fixed_rank", "no_gate", "fine_tune", "ewc"]


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg["input_dim"], cfg["hidden"]),
            nn.GELU(),
            nn.Linear(cfg["hidden"], cfg["classes"]),
        )

    def forward(self, x):
        return self.network(x.float())


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values])


def assign(model, vector):
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(vector[offset : offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()


class FrequentDirections:
    def __init__(self, columns, rows, device):
        self.columns = columns
        self.rows = rows
        self.buffer = torch.empty((0, columns), device=device)
        self.discarded = 0.0

    def append(self, row):
        self.buffer = torch.cat([self.buffer, row.detach()[None]], 0)
        if len(self.buffer) >= 2 * self.rows:
            _, singular, vectors = torch.linalg.svd(self.buffer, full_matrices=False)
            delta = singular[self.rows - 1].square()
            self.discarded += float(delta)
            self.buffer = (singular[: self.rows].square() - delta).clamp_min(0).sqrt()[
                :, None
            ] * vectors[: self.rows]

    def decomposition(self):
        if not len(self.buffer):
            return torch.empty((self.columns, 0), device=self.buffer.device), torch.empty(
                0, device=self.buffer.device
            )
        _, singular, vectors = torch.linalg.svd(self.buffer, full_matrices=False)
        keep = singular > 1e-8
        return vectors[keep].T, singular[keep]


def fit(model, train, validation, cfg, output):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = next(model.parameters()).device
    x = torch.as_tensor(train.data["x"], device=device)
    y = torch.as_tensor(train.data["y"], device=device)
    task = np.asarray(train.data["task"], dtype=int)
    tasks = sorted(np.unique(task).tolist())
    parameters = tuple(model.parameters())
    p = sum(v.numel() for v in parameters)
    rows = min(cfg["maximum_rank"], cfg["memory_bytes"] // (8 * p * 4))
    if rows < 2 and model.method not in ("fine_tune", "ewc"):
        raise ValueError("Memory cap cannot hold a two-row streaming sketch and working basis")
    sketch = FrequentDirections(p, max(2, rows), device)
    Q = torch.empty((p, 0), device=device)
    singular = torch.empty(0, device=device)
    fisher = torch.zeros(p, device=device) if model.method == "ewc" else None
    anchor = flatten(parameters).detach().clone() if model.method == "ewc" else None
    history = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for task_id in tasks:
        indices = np.flatnonzero(task == task_id)
        start = time.perf_counter()
        retentions, ranks, residuals = [], [], []
        for epoch in range(cfg["epochs_per_task"]):
            order = rng.permutation(indices)
            for offset in range(0, len(order), cfg["batch_size"]):
                batch = order[offset : offset + cfg["batch_size"]]
                loss_value = F.cross_entropy(model(x[batch]), y[batch].long())
                current = flatten(parameters)
                if model.method == "ewc":
                    loss_value = (
                        loss_value
                        + 0.5 * cfg["ewc_weight"] * (fisher * (current - anchor).square()).sum()
                    )
                gradients = torch.autograd.grad(loss_value, parameters)
                g = flatten(gradients).detach()
                k = 0
                if Q.shape[1] and model.method not in ("fine_tune", "ewc"):
                    limit = min(cfg["maximum_rank"], Q.shape[1])
                    if model.method == "fixed_rank":
                        k = min(cfg["fixed_rank"], limit)
                    else:
                        k = limit
                        for candidate in range(limit + 1):
                            residual = np.sqrt(
                                (
                                    float(singular[candidate]) ** 2
                                    if candidate < len(singular)
                                    else 0
                                )
                                + sketch.discarded
                            )
                            if residual <= cfg["residual_threshold"]:
                                k = candidate
                                break
                    projected = g - Q[:, :k] @ (Q[:, :k].T @ g)
                    if model.method not in ("no_gate", "fixed_rank"):
                        while (
                            k
                            and projected.norm() / g.norm().clamp_min(1e-12)
                            < cfg["minimum_retention"]
                        ):
                            k -= 1
                            projected = g - Q[:, :k] @ (Q[:, :k].T @ g)
                else:
                    projected = g
                retention = float(projected.norm() / g.norm().clamp_min(1e-12))
                estimate = np.sqrt(
                    (float(singular[k]) ** 2 if k < len(singular) else 0) + sketch.discarded
                )
                residuals.append(cfg["learning_rate"] * estimate * float(g.norm()))
                retentions.append(retention)
                ranks.append(k)
                assign(model, current.detach() - cfg["learning_rate"] * projected)
        sample = rng.choice(indices, min(len(indices), cfg["jacobian_examples"]), replace=False)
        if model.method == "ewc":
            accumulated = torch.zeros_like(fisher)
            for index in sample:
                value = F.cross_entropy(model(x[index : index + 1]), y[index : index + 1].long())
                g = flatten(torch.autograd.grad(value, parameters)).detach()
                accumulated += g.square() / len(sample)
            fisher += accumulated
            anchor = flatten(parameters).detach().clone()
        elif model.method != "fine_tune":
            outputs = min(cfg["jacobian_outputs"], cfg["classes"])
            for index in sample:
                logits = model(x[index : index + 1])[0]
                selected_outputs = rng.choice(cfg["classes"], outputs, replace=False)
                for label in selected_outputs:
                    row = flatten(
                        torch.autograd.grad(logits[int(label)], parameters, retain_graph=True)
                    )
                    sketch.append(row)
            Q, singular = sketch.decomposition()
        protected_bytes = int((sketch.buffer.numel() + Q.numel()) * 4)
        if model.method == "ewc":
            protected_bytes += (fisher.numel() + anchor.numel()) * 4
        history.append(
            {
                "task": task_id,
                "train_loss": float(loss_value.detach()),
                "epoch": len(history) + 1,
                "validation_loss": None,
                "seconds": time.perf_counter() - start,
                "mean_rank": float(np.mean(ranks)),
                "gradient_retention": float(np.mean(retentions)),
                "predicted_first_order_drift": float(np.mean(residuals)),
                "protected_bytes": protected_bytes,
                "working_budget_bytes": cfg["memory_bytes"],
            }
        )
        torch.save({"state": model.state_dict()}, output / f"task_{task_id:04d}.pt")
        torch.save(
            {
                "basis": Q.detach().cpu(),
                "singular": singular.detach().cpu(),
                "discarded": sketch.discarded,
            },
            output / f"sketch_task_{task_id:04d}.pt",
        )
    cfg["training_metrics"] = {
        "protected_bytes": history[-1]["protected_bytes"],
        "gradient_retention": float(np.mean([row["gradient_retention"] for row in history])),
        "mean_rank": float(np.mean([row["mean_rank"] for row in history])),
        "training_seconds": float(sum(row["seconds"] for row in history)),
    }
    cfg["training_method"] = model.method
    torch.save(
        {"state": model.state_dict(), "config": cfg, "method": model.method}, output / "model.pt"
    )
    (output / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )


def loss(model, batch, cfg):
    return F.cross_entropy(model(batch["x"]), batch["y"].long())


def predict(model, batch, cfg, calibration=None):
    return {"logits": model(batch["x"])}


def metrics(prediction, data, cfg):
    result = classification_metrics(prediction["logits"], data["y"])
    result.update(cfg.get("training_metrics", {}))
    tasks = sorted(np.unique(data["task"]).tolist())
    evaluator = Model(cfg, cfg["training_method"])
    matrix = []
    for path in sorted(Path(cfg["_checkpoint_directory"]).glob("task_*.pt")):
        evaluator.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
        with torch.no_grad():
            labels = evaluator(torch.tensor(data["x"])).argmax(-1).numpy()
        matrix.append(
            [
                float(np.mean(labels[data["task"] == task] == data["y"][data["task"] == task]))
                for task in tasks
            ]
        )
    if matrix:
        accuracy = np.asarray(matrix)
        result["average_task_accuracy"] = float(accuracy[-1].mean())
        forgetting = [
            max(accuracy[i:, i]) - accuracy[-1, i]
            for i in range(min(len(tasks), len(accuracy)) - 1)
        ]
        result["forgetting"] = float(np.mean(forgetting)) if forgetting else 0.0
        result["new_task_accuracy"] = float(
            np.mean([accuracy[i, i] for i in range(min(accuracy.shape))])
        )
        result["task_accuracy_matrix"] = accuracy.tolist()
    return result


def inference(model, batch, cfg):
    return model(batch["x"])

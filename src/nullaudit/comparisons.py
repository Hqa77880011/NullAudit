"""Distinct continual-learning controls; algorithm/source notes are in README."""

from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ["prior_rsj", "ogd", "gpm", "lwf", "packnet", "replay", "exact_jacobian"]


def state_bytes(value, seen=None):
    """Count retained tensor payload plus primitive metadata, without tensor aliases."""
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(state_bytes(k, seen) + state_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(state_bytes(v, seen) for v in value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 8 if value is not None else 0


class SampledRSS:
    """Measure process RSS separately from the method's retained protection state."""

    def __enter__(self):
        self.process = psutil.Process()
        self.start = self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        def sample():
            while not self.done.wait(0.002):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def orthogonal_append(basis, row, cap):
    """OGD-GTL Gram-Schmidt update; retain existing old-task directions first."""
    if basis.shape[1] >= cap:
        return basis
    value = row - basis @ (basis.T @ row)
    # A second pass controls roundoff when old directions are nearly dependent.
    value = value - basis @ (basis.T @ value)
    if value.norm() > 1e-7:
        return torch.cat([basis, value[:, None] / value.norm()], 1)
    return basis


def feature_basis(activation, old, energy, cap):
    """GPM residual-activation SVD and cumulative explained-energy selection."""
    residual = activation - old @ (old.T @ activation)
    u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    total = activation.square().sum().clamp_min(1e-12)
    captured = 1 - residual.square().sum() / total
    rank = 0
    while rank < len(singular) and captured < energy and old.shape[1] + rank < cap:
        captured += singular[rank].square() / total
        rank += 1
    if rank:
        return torch.linalg.qr(torch.cat([old, u[:, :rank]], 1), mode="reduced").Q
    return old


def apply_task_mask(model, owners, task_id):
    """PackNet inference includes weights allocated no later than the requested task."""
    from .core import assign, flatten
    vector = flatten(tuple(model.parameters())).detach().clone()
    assign(model, vector * ((owners >= 0) & (owners <= task_id)))
    return vector


def fit(model, train, validation, cfg, output):
    """Fit source-defined controls using only the current task plus declared state."""
    from .core import assign, flatten
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = next(model.parameters()).device
    parameters = tuple(model.parameters())
    p = sum(v.numel() for v in parameters)
    budget = int(cfg["memory_bytes"])
    method = model.method
    task_values = np.asarray(train.data["task"], int)
    tasks = sorted(np.unique(task_values).tolist())
    basis = torch.empty((p, 0), device=device)
    state = {}
    teacher = None
    old_classes = []
    anchors = None
    rsj = torch.empty((0, p), device=device)
    owners = torch.full((p,), -1, dtype=torch.int16, device=device) if method == "packnet" else None
    reservoir_x, reservoir_y = None, None
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
    features = {name: torch.empty((layer.in_features, 0), device=device) for name, layer in layers.items()}
    history = []
    protected_max = 0
    for position, task_id in enumerate(tasks):
        ids = np.flatnonzero(task_values == task_id)
        x = torch.as_tensor(train.data["x"][ids], device=device)
        y = torch.as_tensor(train.data["y"][ids], device=device).long()
        began = time.perf_counter()
        epochs = cfg["epochs_per_task"]
        retentions, losses, gradient_norms, regularizers = [], [], [], []
        initial = flatten(parameters).detach().clone()
        for _ in range(epochs):
            model.train()
            order = rng.permutation(len(x))
            for start in range(0, len(x), cfg["batch_size"]):
                selected = order[start:start + cfg["batch_size"]]
                xb, yb = x[selected], y[selected]
                if method == "replay" and reservoir_x is not None:
                    chosen = rng.choice(len(reservoir_x), min(len(reservoir_x), len(xb)), replace=False)
                    xb = torch.cat([xb, reservoir_x[chosen].to(device)])
                    yb = torch.cat([yb, reservoir_y[chosen].to(device)])
                prediction = model(xb)
                objective = F.cross_entropy(prediction, yb)
                penalty = prediction.sum() * 0
                current = flatten(parameters)
                if method == "lwf" and teacher is not None:
                    temperature = float(cfg.get("distillation_temperature", 2))
                    with torch.no_grad():
                        target = (teacher(xb)[:, old_classes].double() / temperature).log_softmax(-1)
                    penalty = float(cfg.get("distillation_weight", 1)) * temperature ** 2 * F.kl_div(
                        (prediction[:, old_classes].double() / temperature).log_softmax(-1), target,
                        reduction="batchmean", log_target=True).clamp_min(0)
                if method == "prior_rsj" and anchors is not None:
                    penalty = float(cfg.get("sketch_regularization", 1)) / 2 * (rsj @ (current - anchors)).square().sum()
                objective = objective + penalty
                regularizers.append(float(penalty.detach()))
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"Nonfinite {method} loss")
                raw = flatten(torch.autograd.grad(objective, parameters)).detach()
                projected = raw.clone()
                if method in ("ogd", "exact_jacobian"):
                    projected -= basis @ (basis.T @ raw)
                elif method == "gpm" and position:
                    offset = 0
                    for name, parameter in model.named_parameters():
                        length = parameter.numel()
                        layer_name = name.rsplit(".", 1)[0]
                        if name.endswith("weight") and layer_name in features:
                            gradient = projected[offset:offset + length].reshape_as(parameter)
                            u = features[layer_name]
                            gradient.sub_((gradient @ u) @ u.T)
                        elif name.endswith("bias"):
                            projected[offset:offset + length] = 0
                        offset += length
                elif method == "packnet":
                    projected *= (owners < 0)
                if not torch.isfinite(projected).all():
                    raise FloatingPointError(f"Nonfinite {method} gradient")
                gradient_norms.append(float(projected.norm()))
                retentions.append(float(projected.norm() / raw.norm().clamp_min(1e-12)))
                assign(model, current.detach() - cfg["learning_rate"] * projected)
                losses.append(float(objective.detach()))
        model.eval()
        sample = rng.choice(len(x), min(len(x), cfg["jacobian_examples"]), replace=False)
        if method in ("ogd", "exact_jacobian", "prior_rsj"):
            rows = []
            for index in sample:
                logits = model(x[index:index + 1])[0]
                if method == "ogd":
                    labels = [int(y[index])]
                elif method == "prior_rsj":
                    labels = rng.choice(cfg["classes"], min(cfg["jacobian_outputs"], cfg["classes"]), replace=False)
                else:
                    labels = range(cfg["classes"])
                for label in labels:
                    rows.append(flatten(torch.autograd.grad(logits[label], parameters, retain_graph=True)).detach())
            if method == "prior_rsj":
                # Heckel RSJ stacks independently Gaussian-sketched task Jacobians,
                # and recenters the complete quadratic at the latest solution.
                capacity = max(0, (budget - p * 4 - 64) // (p * 4))
                per_task = min(cfg["maximum_rank"], capacity // len(tasks))
                if per_task < 1:
                    raise ValueError("RSJ budget cannot store one sketch row per task and its anchor")
                generator = torch.Generator(device=device).manual_seed(cfg["seed"] + int(task_id))
                matrix = torch.randn(per_task, len(rows), device=device, generator=generator) / np.sqrt(per_task)
                rsj = torch.cat([rsj, matrix @ torch.stack(rows)])
                anchors = flatten(parameters).detach().clone()
                state = {"sketch": rsj, "anchor": anchors}
            else:
                cap = p if method == "exact_jacobian" else max(0, (budget - 64) // (p * 4))
                for row in rows:
                    basis = orthogonal_append(basis, row, cap)
                state = {"basis": basis}
        elif method == "gpm":
            activation = {}
            handles = [layer.register_forward_pre_hook(
                lambda module, inputs, name=name: activation.__setitem__(name, inputs[0].detach().reshape(-1, module.in_features).T))
                for name, layer in layers.items()]
            with torch.no_grad():
                model(x[sample])
            for handle in handles:
                handle.remove()
            remaining = budget - 128
            for name in features:
                width = features[name].shape[0]
                other = sum(value.numel() * 4 for key, value in features.items() if key != name)
                cap = max(0, (remaining - other) // (width * 4))
                features[name] = feature_basis(activation[name], features[name], cfg.get("gpm_energy", 0.95), cap)
            state = {"features": features}
        elif method == "lwf":
            teacher = deepcopy(model).requires_grad_(False).eval()
            old_classes = sorted(set(old_classes) | set(y.tolist()))
            state = {"teacher": teacher.state_dict(), "classes": old_classes}
        elif method == "packnet":
            # Allocate largest-magnitude free weights, prune the rest, then refit
            # only the newly allocated weights while preserving all old weights.
            vector = flatten(parameters).detach().clone()
            free = torch.nonzero(owners < 0, as_tuple=False).flatten()
            keep = max(1, int(len(free) * (1 - cfg.get("packnet_prune_fraction", 0.5))))
            if not len(free):
                raise ValueError("PackNet has exhausted its unallocated weights")
            chosen = free[vector[free].abs().topk(min(keep, len(free))).indices]
            owners[chosen] = int(task_id)
            vector[owners < 0] = 0
            assign(model, vector)
            for _ in range(cfg.get("packnet_refit_steps", 1)):
                objective = F.cross_entropy(model(x[:cfg["batch_size"]]), y[:cfg["batch_size"]])
                gradient = flatten(torch.autograd.grad(objective, parameters)).detach()
                assign(model, flatten(parameters).detach() - cfg["learning_rate"] * gradient * (owners == task_id))
            state = {"owners": owners}
        elif method == "replay":
            old_x = x.detach().cpu() if reservoir_x is None else torch.cat([reservoir_x, x.detach().cpu()])
            old_y = y.detach().cpu() if reservoir_y is None else torch.cat([reservoir_y, y.detach().cpu()])
            each = old_x[0].numel() * old_x.element_size() + 8
            capacity = max(0, (budget - 64) // each)
            if capacity < 1:
                raise ValueError("Replay budget cannot hold one image and label")
            # Uniform sampling over retained old+current records is the explicit
            # bounded-buffer convention; it is not claimed to be exact reservoir sampling.
            selected = rng.choice(len(old_y), min(capacity, len(old_y)), replace=False)
            reservoir_x, reservoir_y = old_x[selected].clone(), old_y[selected].clone()
            state = {"x": reservoir_x, "y": reservoir_y}
        retained = int(state_bytes(state))
        protected_max = max(protected_max, retained)
        if method != "exact_jacobian" and retained > budget:
            raise ValueError(f"{method} protection state {retained} exceeds byte budget {budget}")
        change = float((flatten(parameters).detach() - initial).norm())
        row = dict(task=int(task_id), epoch=position + 1, train_loss=float(np.mean(losses)),
                   validation_loss=None, seconds=time.perf_counter() - began,
                   protected_bytes=retained, working_budget_bytes=budget,
                   gradient_retention=float(np.mean(retentions)), mean_rank=int(basis.shape[1]),
                   gradient_norm=float(np.mean(gradient_norms)), parameter_change_norm=change,
                   regularizer_mean=float(np.mean(regularizers)),
                   oracle_unbounded=method == "exact_jacobian")
        history.append(row)
        torch.save({"state": model.state_dict(), "protection": state}, output / f"task_{task_id:04d}.pt")
        del x, y
    cfg["training_metrics"] = dict(protected_bytes=protected_max,
                                   gradient_retention=float(np.mean([r["gradient_retention"] for r in history])),
                                   mean_rank=float(np.mean([r["mean_rank"] for r in history])),
                                   training_seconds=sum(r["seconds"] for r in history),
                                   oracle_unbounded=method == "exact_jacobian")
    cfg["training_method"] = method
    torch.save({"state": model.state_dict(), "protection": state, "config": cfg, "method": method}, output / "model.pt")
    (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")

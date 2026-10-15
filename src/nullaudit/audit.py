import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import Model, assign, flatten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sketch", required=True)
    parser.add_argument("--old-data", required=True)
    parser.add_argument("--new-data", required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--step-sizes", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sketch = torch.load(args.sketch, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = Model(cfg, payload["method"])
    model.load_state_dict(payload["state"])
    model.eval()
    old = dict(np.load(args.old_data, allow_pickle=False))
    new = dict(np.load(args.new_data, allow_pickle=False))
    old_x = torch.tensor(old["x"][: cfg["batch_size"]])
    new_x = torch.tensor(new["x"][: cfg["batch_size"]])
    new_y = torch.tensor(new["y"][: cfg["batch_size"]])
    original = flatten(tuple(model.parameters())).detach().clone()
    g = flatten(
        torch.autograd.grad(F.cross_entropy(model(new_x), new_y.long()), tuple(model.parameters()))
    ).detach()
    baseline = model(old_x).detach()
    Q, singular = sketch["basis"], sketch["singular"]
    rows = []
    for rank in args.ranks:
        if not 0 <= rank <= Q.shape[1]:
            raise ValueError("Requested rank exceeds the stored sketch")
        projected = g - Q[:, :rank] @ (Q[:, :rank].T @ g)
        residual = np.sqrt(
            (float(singular[rank]) ** 2 if rank < len(singular) else 0) + sketch["discarded"]
        )
        for step in args.step_sizes:
            assign(model, original - step * projected)
            with torch.no_grad():
                drift = torch.linalg.vector_norm(model(old_x) - baseline).item()
            rows.append(
                {
                    "rank": rank,
                    "step_size": step,
                    "realized_old_logit_drift": drift,
                    "predicted_first_order_quantity": step * residual * float(g.norm()),
                    "gradient_retention": float(projected.norm() / g.norm().clamp_min(1e-12)),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

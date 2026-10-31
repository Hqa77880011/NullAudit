"""Small replay-free sequential-task, sketch, and evaluator-only audit checks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import audit, cli, core
from .vision import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(24, 4)).astype(np.float32)
    y = np.tile([0, 1, 2, 3], 6)
    np.savez_compressed(output / "raw.npz", x=x, y=y, task=y // 2,
                        group=np.arange(24), split=np.repeat([0, 1, 2], 8))
    cli.prepare(output / "raw.npz", output / "prepared", 17)
    cfg = json.loads(Path("configs/default.json").read_text())
    cfg.update(input_dim=4, hidden=4, classes=4, epochs=1, epochs_per_task=1,
               maximum_rank=2, fixed_rank=1, memory_bytes=4096,
               batch_size=2, jacobian_examples=2, jacobian_outputs=2)
    data = cli.Arrays(output / "prepared/train.npz")
    batch = {k: torch.as_tensor(v[:2]) for k, v in data.data.items()}
    for method in core.METHODS:
        torch.manual_seed(17)
        model = core.Model(cfg, method)
        initial = [p.detach().clone() for p in model.parameters()]
        loss = core.loss(model, batch, cfg)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        cli.train(cfg, output / "prepared", output / method, method)
        payload = torch.load(output / method / "model.pt", weights_only=False)
        model.load_state_dict(payload["state"])
        assert any(not torch.equal(a, p) for a, p in zip(initial, model.parameters()))
        model.eval()
        restored = core.Model(cfg, method)
        restored.load_state_dict(payload["state"])
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(batch["x"]), restored(batch["x"]), rtol=0, atol=0)
        metrics = cli.evaluate(output / method / "model.pt", output / "prepared/test.npz",
                               output / (method + "-evaluation"))
        assert np.isfinite(metrics["average_task_accuracy"])
        assert metrics["protected_bytes"] <= cfg["memory_bytes"]
    J = torch.randn(8, 4)
    sketch = core.FrequentDirections(4, 2, "cpu")
    for row in J:
        sketch.append(row)
    Q, singular = sketch.decomposition()
    torch.testing.assert_close(Q.T @ Q, torch.eye(Q.shape[1]), rtol=1e-5, atol=1e-5)
    for k in range(Q.shape[1] + 1):
        actual = torch.linalg.matrix_norm(J @ (torch.eye(4) - Q[:, :k] @ Q[:, :k].T), 2)
        bound = np.sqrt((float(singular[k]) ** 2 if k < len(singular) else 0) + sketch.discarded)
        assert actual <= bound + 1e-5
    exact = core.continual_metrics([[0.8, 0.2], [0.5, 0.7]])
    assert np.isclose(exact["average_task_accuracy"], 0.6)
    assert np.isclose(exact["forgetting"], 0.3) and exact["new_task_accuracy"] == 0.75
    assert np.isclose(classification_metrics(np.log([[0.8, 0.2], [0.3, 0.7]]), [0, 1])["ece"], 0.25)
    previous = sys.argv
    sys.argv = ["audit", "--checkpoint", str(output / "nullaudit/model.pt"), "--sketch",
                str(output / "nullaudit/sketch_task_0001.pt"), "--old-data",
                str(output / "prepared/validation.npz"), "--new-data",
                str(output / "prepared/train.npz"), "--ranks", "0", "1", "--step-sizes",
                "0.001", "0.01", "--output", str(output / "audit.json")]
    try:
        audit.main()
    finally:
        sys.argv = previous
    assert len(json.loads((output / "audit.json").read_text())) == 4
    cli.plot([output / "nullaudit", output / "nullaudit-evaluation"], output / "figures")
    print(json.dumps({"check": "passed", "methods": core.METHODS, "audit_rows": 4}))


if __name__ == "__main__":
    main()

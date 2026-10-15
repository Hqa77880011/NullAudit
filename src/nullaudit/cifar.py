import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cifar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", choices=["fine", "coarse"], default="fine")
    parser.add_argument("--clients", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--operator-config")
    args = parser.parse_args()
    training = CIFAR100(args.root, train=True, download=True)
    testing = CIFAR100(args.root, train=False, download=True)
    directory = Path(args.root) / training.base_folder
    with (directory / "train").open("rb") as stream:
        train_metadata = pickle.load(stream, encoding="latin1")
    with (directory / "test").open("rb") as stream:
        test_metadata = pickle.load(stream, encoding="latin1")
    fine = np.r_[train_metadata["fine_labels"], test_metadata["fine_labels"]]
    coarse = np.r_[train_metadata["coarse_labels"], test_metadata["coarse_labels"]]
    x = np.concatenate([training.data, testing.data]).transpose(0, 3, 1, 2).astype(np.float32) / 255
    y = fine if args.labels == "fine" else coarse
    rng = np.random.default_rng(args.seed)
    split = np.full(len(x), 2, dtype=np.int64)
    split[: len(training)] = 0
    for label in np.unique(fine[: len(training)]):
        indices = np.flatnonzero(fine[: len(training)] == label)
        rng.shuffle(indices)
        split[indices[: max(1, len(indices) // 10)]] = 1
    client = np.full(len(x), -1, dtype=np.int64)
    if args.clients:
        if args.clients < 2 or args.alpha <= 0:
            raise ValueError("At least two clients and positive alpha are required")
        for label in np.unique(y):
            indices = np.flatnonzero((y == label) & (split == 0))
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(args.clients, args.alpha))
            boundaries = np.r_[0, np.cumsum(rng.multinomial(len(indices), proportions))]
            for c in range(args.clients):
                client[indices[boundaries[c] : boundaries[c + 1]]] = c
    packed = {
        "x": x,
        "y": y.astype(np.int64),
        "fine_label": fine.astype(np.int64),
        "coarse_label": coarse.astype(np.int64),
        "group": np.arange(len(x)),
        "split": split,
        "task": (fine // args.classes_per_task).astype(np.int64),
    }
    if args.clients:
        packed["client"] = client
    if args.labels == "coarse":
        event = np.zeros((len(x), 100), dtype=np.float32)
        event[np.arange(len(x)), rng.integers(100, size=len(x))] = 1
        packed["event"] = event
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(str(output))
    np.savez_compressed(output, **packed)
    if args.operator_config:
        path = Path(args.operator_config)
        operator = np.zeros((20, 100), dtype=int)
        for fine_class in range(100):
            values = np.unique(coarse[fine == fine_class])
            if len(values) != 1:
                raise ValueError("CIFAR fine class has multiple coarse parents")
            operator[values[0], fine_class] = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(str(path))
        path.write_text(json.dumps({"operator": operator.tolist()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--target", choices=["class", "mask", "count"], default="class")
    parser.add_argument("--imagenet-normalize", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Manifest is empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Every manifest row must have the same fields")
    arrays = {key: [] for key in keys}
    for row in rows:
        for key, value in row.items():
            image_field = (
                key in ("x", "x_pair", "mask", "instances") or key == "y" and args.target == "mask"
            )
            if image_field and isinstance(value, str):
                path = manifest.parent / value
                if path.suffix == ".npy":
                    loaded = np.load(path, allow_pickle=False)
                else:
                    with Image.open(path) as source:
                        if key in ("x", "x_pair"):
                            image = source.convert("RGB" if args.channels == 3 else "L").resize(
                                (args.size, args.size), Image.Resampling.BILINEAR
                            )
                            loaded = np.asarray(image, dtype=np.float32) / 255
                            loaded = loaded[None] if loaded.ndim == 2 else loaded.transpose(2, 0, 1)
                            if args.imagenet_normalize:
                                if args.channels != 3:
                                    raise ValueError("ImageNet normalization requires RGB")
                                loaded = (
                                    loaded - np.array([0.485, 0.456, 0.406])[:, None, None]
                                ) / np.array([0.229, 0.224, 0.225])[:, None, None]
                        else:
                            image = source.resize((args.size, args.size), Image.Resampling.NEAREST)
                            loaded = np.asarray(image)
                            if loaded.ndim != 2:
                                raise ValueError("Masks must have one channel")
                            loaded = (
                                loaded.astype(np.int64)
                                if key == "instances"
                                else (loaded > 0).astype(np.float32)[None]
                            )
                arrays[key].append(loaded)
            elif key == "split":
                arrays[key].append({"train": 0, "validation": 1, "test": 2}.get(value, value))
            elif isinstance(value, str) and key != "group" and value.endswith(".npy"):
                arrays[key].append(np.load(manifest.parent / value, allow_pickle=False))
            else:
                arrays[key].append(value)
    packed = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype == object for value in packed.values()):
        raise ValueError("Ragged/object arrays are unsupported; use one fixed shape per dataset")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(str(output))
    np.savez_compressed(output, **packed)


if __name__ == "__main__":
    main()

# NullAudit

Implementation of **Residual-guided adaptive function-space projection for replay-free continual learning**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The CIFAR adapter downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), reserves 10% of training examples for validation, preserves the test set and records sequential class-task IDs.

```bash
python -m nullaudit.cifar --root data/cifar --download --output data/raw.npz --seed 17
python -m nullaudit prepare --source data/raw.npz --output data/prepared --seed 17
```

## Training

Use the same `configs/default.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m nullaudit train --config configs/default.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m nullaudit evaluate --checkpoint runs/main/nullaudit/model.pt --data data/prepared/test.npz --output results/nullaudit
```

## Comparisons and ablations

`suite` trains the implemented methods on the same prepared split. Use `--methods` to select a subset, then evaluate each method checkpoint with the evaluation command above.

```bash
python -m nullaudit suite --config configs/default.json --data data/prepared --output runs/comparison
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m nullaudit plot --inputs runs/main/nullaudit results/nullaudit --output figures
```

## Implemented scope and memory conventions

This is an implementation written from the manuscript, separate from its original experiment artifacts. The model is a two-layer MLP. Tasks arrive in increasing declared `task` order. The learner moves only the current task's examples to its compute device, then drops those tensors at the task boundary; its protected state consists of a streaming Frequent-Directions sketch/basis (or EWC diagonal/anchor), never replay examples. The offline NPZ dataset remains the input provider in host memory and is excluded from method-state memory.

| Paper module / endpoint | Entry | Command |
| --- | --- | --- |
| Streaming Jacobian FD sketch and residual energy | `core.FrequentDirections` | train command above |
| Residual-based rank selection and gradient-retention gate | `core.fit` | same train command |
| Average task accuracy, maximum-past forgetting, new-task accuracy and accuracy matrix | `core.continual_metrics`, `core.metrics` | evaluate command above |
| Stored tensor bytes, retention, rank, update time and predicted first-order quantity | `core.fit` history / config | same train/evaluate commands |
| Evaluator-only realized drift versus rank/step size | `audit` | command below |
| Fixed-rank, no-gate, fine-tuning and EWC controls; plots | `core.METHODS`, `cli.plot` | suite / plot commands above |

```bash
python -m nullaudit.audit --checkpoint runs/main/nullaudit/model.pt --sketch runs/main/nullaudit/sketch_task_0009.pt --old-data data/prepared/validation.npz --new-data data/prepared/train.npz --ranks 0 4 8 --step-sizes 0.001 0.01 --output results/drift_audit.json
```

Select a sketch filename and ranks that actually exist in the run. `old-data` and `new-data` must be evaluator exports for the declared old/new tasks; they are never consumed by the training controller. The audit restores the same original model before each update and reports actual old-logit drift and the residual-derived first-order quantity, not a verified global bound. It must be repeated at multiple step sizes to inspect curvature.

`maximum_rank` and `memory_bytes` allocate sketch rows using an eight-buffer float32 working allowance. `protected_bytes` records retained sketch/basis/singular-value tensors, the discarded-energy scalar and, for EWC, the diagonal/anchor tensors. It is not an OS peak-memory measurement or an exact SVD workspace cap. The model, input provider, gradients and Python runtime are excluded; RSS is separately reported during evaluation. Dense SVD workspace, task metadata and full system peak memory have not been profiled. The default 64 MiB is therefore a protected-state/analytic-working-budget setting, not an empirically audited total process limit.

Accuracy metrics are fractions. Forgetting is maximum accuracy after learning each old task minus final accuracy, averaged over old tasks. New-task accuracy averages the diagonal of the task accuracy matrix. ECE uses integer-indexed bins so confidences at floating-point boundaries are counted exactly once. The CIFAR adapter preserves released test images; no test labels tune ranks or gates.

The supplied manuscript does not release its exact Split-CIFAR backbone or original experimental artifacts. The common-backbone comparisons use this declared MLP, class order, cross-entropy, batch schedule and seed. The nearest cited prior, Heckel (2022), uses **regularization with a sketched Jacobian (RSJ)**, rather than hard projection. `prior_rsj` implements Gaussian `S J` rows and `lambda/2 * ||K(theta-anchor)||^2`, recentering the accumulated quadratic at the latest task solution. Its Jacobian example/output sampling matches NullAudit. It is separate from the existing `fixed_rank` hard-projection control.

## Source-defined comparison methods

These are independent implementations after checking the original methods; upstream code and licenses have not been vendored or relicensed.

| Method | Mechanism | Primary source / official implementation |
|---|---|---|
| `prior_rsj` | Gaussian task-Jacobian sketch, recentered quadratic regularization | [Heckel 2022, Sec.3](https://proceedings.mlr.press/v151/heckel22a/heckel22a.pdf) |
| `ogd` | Ground-truth-logit gradients, incremental Gram-Schmidt and projection | [OGD-GTL](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf) |
| `gpm` | Activation residual SVD and layer-input gradient projection | [official GPM](https://github.com/sahagobinda/GPM/blob/main/main_pmnist.py) |
| `lwf` | Frozen previous network, old-class temperature distillation on current images | [official LwF](https://github.com/lizhitwo/LearningWithoutForgetting) |
| `packnet` | Magnitude pruning, persistent ownership, old-weight freezing and mask refitting | [official PackNet](https://github.com/arunmallya/packnet/blob/master/src/prune.py) |
| `replay` | Explicit bounded old-example buffer mixed into current batches | Declared uniform retained-plus-current replay control |
| `exact_jacobian` | Every sampled-example logit gradient, unbounded basis | Manuscript diagnostic oracle |

GPM uses this shared MLP's layer activations and freezes biases after task one. PackNet uses per-parameter ownership, default prune fraction 0.5 and one refit step; its requested-task mask makes this a **task-aware** comparison. Replay uniformly resamples retained-plus-current records rather than claiming exact reservoir sampling. LwF explicitly counts its complete frozen teacher and class metadata as nonzero protected state. The exact-Jacobian oracle bypasses the state cap and is not a fixed-budget competitor. Full multi-seed CIFAR performance and the manuscript's numerical results remain unverified.

## Equal-byte frontier and measured memory

```bash
python -m nullaudit.frontier --config configs/default.json --data data/prepared --methods nullaudit prior_rsj ogd gpm lwf packnet replay --budgets 16777216 33554432 67108864 --output runs/equal-byte
```

`frontier.csv/json/png` use identical **byte caps** and plot actual retained bytes; unused capacity is not padded. Infeasible configurations remain marked `infeasible`. New controls' retained payload counts tensors, deduplicated aliases and serialized primitive metadata. Python object/allocator overhead, the common model, provider arrays and optimizer are outside this payload definition. `process_peak_rss_bytes_sampled` is measured separately every 2 ms during training and includes those process allocations; it may miss shorter spikes. The existing eight-buffer sketch allowance remains analytic, not an empirical RSS cap.

## Prompt and vision-language paths

Install `python -m pip install -e ".[prompts]"`. The independent path uses an actual locally configured Hugging Face CLIP vision/text backbone, class-ordered tokenizer IDs (`C x sequence_length` NPY) and optionally a local pretrained CLIP state_dict. It automatically downloads no weights.

```bash
python -m nullaudit.prompts --config data/clip-config.json --text-tokens data/class-tokens.npy --backbone-state data/clip-state.pt --data data/prepared/train.npz --method pgp --epochs 5 --output runs/pgp
python -m nullaudit.prompts --config data/clip-config.json --text-tokens data/class-tokens.npy --backbone-state data/clip-state.pt --data data/prepared/train.npz --method vpt_ns --epochs 5 --output runs/vpt-ns
python -m nullaudit.prompts --config data/clip-config.json --text-tokens data/class-tokens.npy --backbone-state data/clip-state.pt --data data/prepared/train.npz --method dmnsp --epochs 5 --output runs/dmnsp
```

`pgp` learns key/value prefixes and projects gradients against stored query-feature directions, following [official PGP](https://github.com/JingyangQiao/prompt-gradient-projection/tree/main/dualprompt-pgp). The declared adaptation uses one shared prefix per layer without DualPrompt's retrieval pool. `vpt_ns` inserts deep prompt tokens, collects both `Q W_k` and original-to-prompt attention covariances, and projects the **optimizer update** on both sides, including the times-threshold/Frobenius convention from [official VPT-NS](https://github.com/zugexiaodui/VPTinNSforCL/blob/main/train_eval.py). `dmnsp` adds residual bottleneck adapters to both actual CLIP modalities, with token-derived class prototypes. Only visual gradients receive multi-layer null projections and `exp(-top-10%-similarity)*30` coefficients from [official DMNSP](https://github.com/RL-VIG/DMNSP/blob/master/continual_clip/models.py); text adapters train normally. For small dimensions, top-k is at least one. These are explicit backbone/mechanism adaptations, not upstream pretrained recipe reproductions. Trainable method parameters and projection state are separate outputs; this different backbone is not conflated with the MLP equal-byte frontier.

## Additional functional verification

```bash
python -m nullaudit.verify_comparisons --output runs/comparisons-check
python -m nullaudit.verify_prompts --output runs/prompts-check
```

The comparison check reads generated **official CIFAR Python-release pickle structure**, preserves the test role, reserves validation from training and verifies task IDs. The adapter output remains `N,3,32,32`; only this cheap state check pools those images to three channels. Local releases do not need download. Class order is seeded/random by default (`--ordered-classes` selects ascending order). Pickle input must be a trusted official release or explicit local fixture. All seven new controls run through training, active regularization/projection, critical first/last-layer updates, 1e-6 save/load, held-out evaluation, PackNet old-weight invariance and OGD/GPM hand cases. Ten equal-cap configurations include nine computed and one correctly reported infeasible tiny-cap NullAudit run, with every computed retained state inside its cap.

The prompt check uses an actual random two-layer CLIP vision/text configuration, two tasks and separate held-out generated images. Nonempty old-task memory exercises PGP prefix projection, both VPT consistency matrices, and DMNSP visual/text adapters; critical nonzero gradients, updates, held-out task matrices, a hand null projector and 1e-6 reload equality are checked. Random token/weight fixtures establish functionality only. On this host set process-local `USE_TF=0`, `USE_FLAX=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL`. Optional dependencies for this pass are isolated in ignored `.deps/`.

## Functional verification

```bash
python -m nullaudit.verify --output runs/check
```

Actually run on 2026-11-01 with Python 3.13.5, torch 2.7.1+cu118 and numpy 2.1.3, CPU, seed 17, one torch thread. The check prepares a generated two-task NPZ, runs all five methods for one epoch per task, checks finite loss/gradients, model updates, exact restored-checkpoint logits and the retained-state budget, then runs a separate four-row rank/step evaluator audit and plots. It checks FD-basis orthonormality and the residual upper bound against an explicit small Jacobian. A hand accuracy matrix [[0.8, 0.2], [0.5, 0.7]] yields final average 0.6, forgetting 0.3 and new-task accuracy 0.75; the ECE boundary fixture gives 0.25. Generated examples validate functionality, not paper performance. No full CIFAR experiment was run.

Use process-local `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL` on this host to select sequential MKL when starting Python.

## Native DualPrompt-PGP pool

The earlier `nullaudit.prompts --method pgp` command remains the declared shared-prefix CLIP adaptation. `nullaudit.dualprompt` additionally implements the native ViT/DualPrompt structure: frozen ViT image encoder and unprompted CLS retrieval features, general key/value prefixes at layers 0/1, learned expert pool at layers 2/3/4, cosine-key top-k retrieval with batchwise voting, task-specific expert ranges during training, shared-pool initialization from the preceding task, a trainable class head, and the key-pulling objective. Inference retrieves experts without a task label. PGP projects **key-prefix** gradients while leaving value-prefix gradients unchanged, matching the original DualPrompt-PGP engine. Optimizer state resets at task boundaries.

```bash
python -m nullaudit.dualprompt --config configs/dualprompt_pgp_cifar100.json --data data/prepared/train.npz --backbone-state data/torchvision-vit-b16-state.pt --output runs/dualprompt-pgp
python -m nullaudit.dualprompt --config configs/dualprompt_pgp_cifar100.json --data data/prepared/test.npz --checkpoint runs/dualprompt-pgp/model.pt --output runs/dualprompt-pgp/test.npz
python -m nullaudit.verify_dualprompt --output runs/native-pgp-check
```

The executable config recovers the [official CIFAR-100/10-task recipe](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/configs/cifar100_10_dualprompt_pgp.py): ViT-B/16 at 224, pool size 10, top-k 1, prompt length 5, Adam LR 0.03, batch 24, 20 epochs/task, zero decay, gradient clipping 1, smoothing 0.1 and seed 59. [Pool retrieval](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/prompt.py), [prefix attachment](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/vision_transformer.py), [projection and threshold 0.50](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/engine.py) were read directly. The independent implementation uses torchvision ViT state dictionaries, ImageNet normalization and feature-axis query-covariance SVD; it does not copy the upstream hard-coded 768-sample/PCA memory collector. Smoothing is restricted to current-task classes so masked classes cannot receive invalid infinite-loss smoothing mass. No upstream code is vendored or relicensed. Supply a compatible local pretrained state; omitting it deliberately uses random weights.

CPU one-thread verification on 2026-11-01 passed a five-layer, width-16, two-task generated fixture. General prefixes, expert prefixes, pool keys and classifier received finite nonzero gradients and updated. The task-one inactive expert remained gradient-free; expert copying, task masks, cosine retrieval without task IDs, old-feature key-gradient orthogonality and untouched value gradients passed. Full state save/load gave exactly identical held-out logits; two task losses were 0.57955 and 0.88165. `memory_threshold=0.99` in the tiny check keeps nonempty constraints, while the recovered config retains source threshold 0.50. This closes the missing DualPrompt-pool mechanism; the original NullAudit backbone/artifacts and pretrained full-CIFAR performance remain unavailable/unrun under the no-full-training constraint.

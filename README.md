# Action-Free Values

Offline goal-conditioned RL when only a small fraction of trajectories have action labels.

The value of reaching a goal can be learned from states alone. Only the parts that turn the value into actions need labels, and those parts never see the goal:

- **Cube (manipulation):** a bilinear value V(s, g) = φ(s)ᵀψ(g) is trained on every row with expectile TD (as in GCIVL). On labeled rows, a one-step model μ(s, a) ≈ E[φ(s′) | s, a] is fit, written as φ(s) + Δ(s, a). Together they give Q(s, a, g) = r + γ μ(s, a)ᵀψ(g) for every goal. The actor is trained with DDPG+BC on that Q, using labeled rows only.
- **AntMaze (navigation):** a quasimetric value (QRL) is trained on every row. The actor is trained with AWR on labeled rows, and only for goals at most 25 steps ahead. At evaluation, far goals are replaced by waypoints chosen among dataset states using the learned distance.

The code follows the layout of [OGBench](https://github.com/seohongpark/ogbench) and uses its environments and datasets (`pip install ogbench`). `impls/agents/` and `impls/utils/` are OGBench's, with the additions listed below.

## Installation

```bash
pip install -r impls/requirements.txt
```

Set `OGBENCH_DATA_DIR` to read datasets from somewhere other than `~/.ogbench/data`.

## Usage

All commands run from `impls/`.

```bash
# 1. Keep the action labels of 10% of the trajectories.
python ../data_gen_scripts/make_label_masks.py --env_name=cube-single-play-v0 --p=0.1 --label_seed=0 \
    --save_path=masks/cube-single-play-v0_p0.1_s0.npz

# 2. Train AFV.
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py \
    --label_mask_path=masks/cube-single-play-v0_p0.1_s0.npz
```

`impls/hyperparameters.sh` lists the exact command for every result in the paper.

## What is added to OGBench

| File | Addition |
|---|---|
| `agents/afv.py` | The AFV agent (new). |
| `agents/idm.py` | Inverse dynamics model that fills in missing actions, the first stage of the SS-ORL baseline (new). |
| `agents/gciql.py`, `gcivl.py`, `hiql.py`, `qrl.py` | With a label mask, the losses that read actions average over labeled rows only. `qrl.py` also has ablation switches for its dynamics model. |
| `utils/datasets.py` | `actor_offset_max`: draw actor goals from the next k states. `actor_q_reward`: rewards for the actor goal. |
| `utils/waypoint.py` | Waypoint selection at evaluation time (new). |
| `main.py` | `--label_mask_path`, `--impute_path`, `--corrupt_labels`, `--eval_only`, `--waypoint_*`. |
| `data_gen_scripts/` | Label masks and noisy-observation datasets. |

## Paper

LaTeX source in `paper/`.

## License

MIT. `impls/agents/` and `impls/utils/` are derived from OGBench (Copyright (c) 2024 OGBench Authors); see `LICENSE`.

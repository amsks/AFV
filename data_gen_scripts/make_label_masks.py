"""Pick a fraction p of trajectories whose actions stay labeled, and save a per-row mask.

OGBench compact datasets end each trajectory with an extra observation-only row (valids == 0), so trajectories are
counted from valid terminal rows. Exactly ceil(p * num_trajectories) whole trajectories are labeled.

Usage (from impls/):
    python ../data_gen_scripts/make_label_masks.py --env_name=cube-single-play-v0 --p=0.1 --label_seed=0 \
        --save_path=masks/cube-single-play-v0_p0.1_s0.npz
"""

import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'impls'))
from utils.env_utils import make_env_and_datasets  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cube-single-play-v0', 'Environment (dataset) name.')
flags.DEFINE_float('p', 0.1, 'Fraction of trajectories that keep their action labels.')
flags.DEFINE_integer('label_seed', 0, 'Seed for choosing the labeled trajectories.')
flags.DEFINE_string('save_path', None, 'Output .npz path (key: mask).')


def trajectory_segments(terminals, valids):
    """Start/end row of every real trajectory (end row inclusive)."""
    ends = np.nonzero(terminals > 0)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    real = valids[ends] > 0
    return starts[real], ends[real]


def select_labeled_trajs(terminals, valids, p, label_seed):
    """Per-row int8 mask, 1 on the rows of ceil(p * num_trajectories) randomly chosen trajectories."""
    starts, ends = trajectory_segments(terminals, valids)
    num_labeled = int(np.ceil(p * len(starts)))
    chosen = np.random.default_rng(label_seed).choice(len(starts), size=num_labeled, replace=False)
    mask = np.zeros(len(terminals), dtype=np.int8)
    for j in chosen:
        mask[starts[j] : ends[j] + 1] = 1
    return mask


def main(_):
    _, train_dataset, _ = make_env_and_datasets(FLAGS.env_name)
    mask = select_labeled_trajs(
        np.asarray(train_dataset['terminals']), np.asarray(train_dataset['valids']), FLAGS.p, FLAGS.label_seed
    )
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.save_path)), exist_ok=True)
    np.savez(FLAGS.save_path, mask=mask)
    print(f'Saved {FLAGS.save_path}: {int(mask.sum())} of {len(mask)} rows labeled.')


if __name__ == '__main__':
    app.run(main)

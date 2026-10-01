"""Add Gaussian observation noise to an OGBench dataset (training data only; the evaluation environment stays clean).

Each stored observation row gets noise sigma * std_per_dim, where std_per_dim is the per-dimension standard deviation of
the clean training observations. Since compact datasets store s' as the next row, the logged trajectory stays
self-consistent. All other keys are copied unchanged, so label masks remain valid.

The new dataset replaces the dataset-type token with the noise level, e.g. cube-single-play-v0 with sigma=0.1 becomes
cube-single-sig0p1-v0, which OGBench still resolves to the cube-single-v0 environment.

Usage:
    python add_obs_noise.py --dataset=cube-single-play-v0 --sigmas=0.1,0.3,1.0
"""

import os

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cube-single-play-v0', 'Source dataset name.')
flags.DEFINE_string('sigmas', '0.1,0.3,1.0', 'Comma-separated noise levels, in units of per-dimension std.')
flags.DEFINE_string('data_dir', os.path.expanduser('~/.ogbench/data'), 'OGBench dataset directory.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def noisy_name(dataset, sigma):
    splits = dataset.split('-')
    return '-'.join(splits[:-2] + ['sig' + f'{sigma:.1f}'.replace('.', 'p')] + splits[-1:])


def main(_):
    with np.load(os.path.join(FLAGS.data_dir, f'{FLAGS.dataset}.npz')) as f:
        std = f['observations'].astype(np.float64).std(axis=0, keepdims=True)

    for sigma in [float(s) for s in FLAGS.sigmas.split(',')]:
        name = noisy_name(FLAGS.dataset, sigma)
        for part, suffix in enumerate(['', '-val']):
            with np.load(os.path.join(FLAGS.data_dir, f'{FLAGS.dataset}{suffix}.npz')) as f:
                data = {k: f[k][...] for k in f.files}
            rng = np.random.default_rng([FLAGS.seed, int(round(1000 * sigma)), part])
            obs = data['observations']
            data['observations'] = (obs + sigma * std * rng.standard_normal(obs.shape)).astype(obs.dtype)
            save_path = os.path.join(FLAGS.data_dir, f'{name}{suffix}.npz')
            np.savez_compressed(save_path, **data)
            print(f'Saved {save_path}')


if __name__ == '__main__':
    app.run(main)

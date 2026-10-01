import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.waypoint import WaypointPolicy, bank_from_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_integer('eval_only', 0, 'Only evaluate the agent restored from restore_path/restore_epoch.')

# Label budget.
flags.DEFINE_string('label_mask_path', None, 'Label mask (.npz, key: mask). Action losses use only rows with mask 1.')
flags.DEFINE_string(
    'impute_path', None, 'Imputed actions (.npz, key: actions) to train on instead of the dataset actions.'
)
flags.DEFINE_string('corrupt_labels', None, "Corrupt the labeled actions: 'shuffle' or 'uniform'.")

# Evaluation-time waypoints (QRL only).
flags.DEFINE_integer('waypoint_bank_size', 0, 'Number of dataset states in the waypoint bank (0: no waypoints).')
flags.DEFINE_float('waypoint_k', 25, 'Goal horizon of the actor, in steps.')
flags.DEFINE_float('waypoint_band_lo', 0.5, 'Waypoints are chosen with d(s, w) in [band_lo * k, band_hi * k].')
flags.DEFINE_float('waypoint_band_hi', 1.5, 'See waypoint_band_lo.')
flags.DEFINE_float('waypoint_reach', 5, 'Switch to a new waypoint once d(s, w) falls to this.')
flags.DEFINE_integer('waypoint_recompute_every', 25, 'Re-plan at least this often.')
flags.DEFINE_float('waypoint_multihop_radius', 0, 'If > 0, rank waypoints by bank-graph cost to the goal.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='AFV', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    train_dataset = dict(train_dataset)
    if FLAGS.label_mask_path is not None:
        mask = np.load(FLAGS.label_mask_path)['mask'].astype(np.int8)
        assert len(mask) == len(train_dataset['actions'])
        train_dataset['labeled_mask'] = mask
        if FLAGS.corrupt_labels is not None:
            # Break the (s, a, s') correspondence on labeled rows; unlabeled rows are untouched.
            rng = np.random.default_rng(FLAGS.seed)
            labeled = np.nonzero(mask)[0]
            actions = train_dataset['actions'].copy()
            if FLAGS.corrupt_labels == 'shuffle':
                actions[labeled] = actions[labeled][rng.permutation(len(labeled))]
            else:
                actions[labeled] = rng.uniform(-1.0, 1.0, size=actions[labeled].shape)
            train_dataset['actions'] = actions
    if FLAGS.impute_path is not None:
        # SS-ORL: every row now has an action (real on labeled rows, imputed elsewhere), so no mask.
        train_dataset['actions'] = np.load(FLAGS.impute_path)['actions'].astype(np.float32)
        assert len(train_dataset['actions']) == len(train_dataset['observations'])
        train_dataset.pop('labeled_mask', None)
    if 'labeled_mask' in train_dataset:
        # Validation rows carry no mask; the masked losses are not comparable there.
        val_dataset = None

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    def evaluate_all(agent):
        if FLAGS.eval_on_cpu:
            agent = jax.device_put(agent, device=jax.devices('cpu')[0])
        if FLAGS.waypoint_bank_size > 0:
            agent = WaypointPolicy(
                agent,
                bank_from_dataset(train_dataset.dataset, FLAGS.waypoint_bank_size),
                k=FLAGS.waypoint_k,
                band=(FLAGS.waypoint_band_lo, FLAGS.waypoint_band_hi),
                reach=FLAGS.waypoint_reach,
                recompute_every=FLAGS.waypoint_recompute_every,
                multihop_radius=FLAGS.waypoint_multihop_radius,
            )
        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
        for task_id in tqdm.trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=env,
                task_id=task_id,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update({f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)
        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        if FLAGS.video_episodes > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video
        return eval_metrics

    if FLAGS.eval_only:
        eval_metrics = evaluate_all(agent)
        wandb.log(eval_metrics, step=FLAGS.restore_epoch)
        eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
        eval_logger.log(eval_metrics, step=FLAGS.restore_epoch)
        eval_logger.close()
        return

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval > 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = evaluate_all(agent)
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()

    if config['agent_name'] == 'idm':
        # First stage of SS-ORL: save actions for every row, to be passed to the second stage as --impute_path.
        actions = agent.impute(train_dataset.dataset, train_dataset.dataset['labeled_mask'])
        np.savez(os.path.join(FLAGS.save_dir, 'imputed_actions.npz'), actions=actions)


if __name__ == '__main__':
    app.run(main)

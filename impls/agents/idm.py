from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class IDMAgent(flax.struct.PyTreeNode):
    """Inverse dynamics model a = f(s, s'), trained on labeled rows.

    First stage of the SS-ORL baseline: `impute` fills in the actions of unlabeled rows, and a standard agent is then
    trained on the completed dataset (main.py: `--impute_path`).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        inputs = jnp.concatenate([batch['observations'], batch['next_observations']], axis=-1)
        pred = self.network.select('idm')(inputs, params=grad_params)
        per_row = ((pred - batch['actions']) ** 2).sum(-1)
        mask = batch['labeled_mask'].astype(jnp.float32) if 'labeled_mask' in batch else jnp.ones_like(per_row)
        loss = (per_row * mask).sum() / (mask.sum() + 1e-6)
        return loss, {'idm_loss': loss}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda gp: self.total_loss(batch, gp, rng=rng))
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def predict(self, observations, next_observations):
        inputs = jnp.concatenate([observations, next_observations], axis=-1)
        return jnp.clip(self.network.select('idm')(inputs), -1, 1)

    def impute(self, dataset, labeled_mask, chunk=4096):
        """Actions for every row: the real action on labeled rows, the predicted one elsewhere."""
        obs = np.asarray(dataset['observations'])
        next_obs = np.concatenate([obs[1:], obs[-1:]])  # Compact dataset: s' is the next row.
        pred = np.concatenate(
            [np.asarray(self.predict(obs[i : i + chunk], next_obs[i : i + chunk])) for i in range(0, len(obs), chunk)]
        )
        return np.where(labeled_mask[:, None] > 0, np.asarray(dataset['actions']), pred).astype(np.float32)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        idm_def = MLP(
            (*config['hidden_dims'], ex_actions.shape[-1]), activate_final=False, layer_norm=config['layer_norm']
        )
        network_def = ModuleDict({'idm': idm_def})
        params = network_def.init(init_rng, idm=(jnp.concatenate([ex_observations, ex_observations], axis=-1),))[
            'params'
        ]
        network = TrainState.create(network_def, params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='idm',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            hidden_dims=(512, 512),  # Hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor (only used by the goal sampler).
            discrete=False,  # Whether the action space is discrete.
            # Dataset hyperparameters (the goals are unused).
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config

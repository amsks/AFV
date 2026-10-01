import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCBilinearValue, ensemblize


def masked_mean(x, mask):
    return (x * mask).sum() / (mask.sum() + 1e-6)


class AFVAgent(flax.struct.PyTreeNode):
    """Action-free values (AFV) agent.

    1. Value (all rows, no actions): V(s, g) = phi(s)^T psi(g) / sqrt(d), trained with expectile TD as in GCIVL.
    2. One-step model (labeled rows only): mu(s, a) ~ E[phi(s') | s, a]. With `mu_residual`, mu(s, a) = phi(s) +
       Delta(s, a). An ensemble of `mu_ensemble_size` models gives Q-hat a lower confidence bound.
    3. Actor (labeled rows only): DDPG+BC on Q-hat(s, a, g) = r(s, g) + gamma * m(s, g) * mu(s, a)^T psi(g) / sqrt(d).

    Ablations from the paper:
    - actor_loss='awr': AWR with the value advantage V(s', g) - V(s, g); no one-step model.
    - grounding='scalar_q' (+ actor_loss='q_awr'): one head q(s, g, a) regressed on r + gamma * m * V(s', g), with no
      phi/psi factorization.
    - actor_q_unmasked, mu_full_labels, actor_full_labels: which rows each loss sees.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_bounds(self):
        """Range of V implied by the sparse reward."""
        bound = 1.0 / (1.0 - self.config['discount'])
        return (-bound, 0.0) if self.config['gc_negative'] else (0.0, bound)

    def clip_value(self, v):
        if not self.config['clip_value_target']:
            return v
        return jnp.clip(v, *self.value_bounds())

    def value_loss(self, batch, grad_params):
        """Expectile TD on the bilinear value. Uses every row and no actions."""
        next_v = self.clip_value(self.network.select('target_value')(batch['next_observations'], batch['value_goals']))
        v_t = self.clip_value(self.network.select('target_value')(batch['observations'], batch['value_goals']))
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        adv = q - v_t
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(adv, q - v, self.config['expectile']).mean()
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean(), 'adv': adv.mean()}

    def target_features(self, observations, goals):
        """phi(observations) and psi(goals) from the target value network, without gradients."""
        _, phi, psi = self.network.select('target_value')(observations, goals, info=True)
        return jax.lax.stop_gradient(phi), jax.lax.stop_gradient(psi)

    def mu(self, observations, actions, params=None):
        """One-step model: [K, B, d] (or [B, d] when K=1)."""
        out = self.network.select('mu')(jnp.concatenate([observations, actions], axis=-1), params=params)
        if self.config['mu_residual']:
            phi_s, _ = self.target_features(observations, observations)
            out = phi_s + out
        return out

    def mu_loss(self, batch, grad_params):
        """Regress mu(s, a) onto phi(s') on labeled rows."""
        phi_next, _ = self.target_features(batch['next_observations'], batch['next_observations'])
        mu = self.mu(batch['observations'], batch['actions'], params=grad_params)
        per_row = ((mu - phi_next) ** 2).sum(-1) / self.config['rep_dim']
        if self.config['mu_ensemble_size'] > 1:
            per_row = per_row.mean(0)
        mask = batch['labeled_mask'].astype(jnp.float32)
        if self.config['mu_full_labels']:
            mask = jnp.ones_like(mask)
        mu_loss = masked_mean(per_row, mask)
        return mu_loss, {'mu_loss': mu_loss}

    def q_reg_loss(self, batch, grad_params):
        """grounding='scalar_q': regress q(s, g, a) onto r + gamma * m * V(s', g) on labeled rows."""
        next_v = self.clip_value(self.network.select('target_value')(batch['next_observations'], batch['value_goals']))
        target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        sga = jnp.concatenate([batch['observations'], batch['value_goals'], batch['actions']], axis=-1)
        q = self.network.select('q')(sga, params=grad_params).squeeze(-1)
        diff = q - jax.lax.stop_gradient(target)
        weight = jnp.where(diff < 0, self.config['q_expectile'], 1.0 - self.config['q_expectile'])
        per_row = 2.0 * weight * diff**2
        if self.config['mu_ensemble_size'] > 1:
            per_row = per_row.mean(0)
        q_loss = masked_mean(per_row, batch['labeled_mask'].astype(jnp.float32))
        return q_loss, {'q_loss': q_loss}

    def qhat(self, batch, actions):
        """Q-hat(s, a, g) at the actor goal, per ensemble member: [K, B] (or [B])."""
        if self.config['grounding'] == 'scalar_q':
            sga = jnp.concatenate([batch['observations'], batch['actor_goals'], actions], axis=-1)
            return self.network.select('q')(sga).squeeze(-1)
        _, psi = self.target_features(batch['observations'], batch['actor_goals'])
        mu = self.mu(batch['observations'], actions)
        if self.config['actor_q_reward']:
            r, m = batch['actor_rewards'], batch['actor_masks']
        else:
            r, m = 0.0, 1.0
        return self.clip_value(r + self.config['discount'] * m * (mu * psi).sum(-1) / jnp.sqrt(self.config['rep_dim']))

    def actor_loss(self, batch, grad_params, rng):
        mask = batch['labeled_mask'].astype(jnp.float32)
        if self.config['actor_full_labels']:
            mask = jnp.ones_like(mask)
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        if self.config['actor_loss'] in ('awr', 'q_awr'):
            v = self.network.select('value')(batch['observations'], batch['actor_goals'])
            if self.config['actor_loss'] == 'awr':
                # Value advantage; the one-step model is not used.
                adv = self.network.select('value')(batch['next_observations'], batch['actor_goals']) - v
                exp_a = jnp.exp(adv * self.config['awr_temperature'])
            else:
                # Advantage of the q head at the dataset action, whitened over labeled rows.
                q = self.qhat(batch, batch['actions'])
                if self.config['mu_ensemble_size'] > 1:
                    q = q.mean(0)
                adv = q - v
                adv_m = masked_mean(adv, mask)
                adv_w = (adv - adv_m) / jnp.sqrt(masked_mean((adv - adv_m) ** 2, mask) + 1e-8)
                exp_a = jax.lax.stop_gradient(jnp.exp(adv_w * self.config['awr_temperature']))
            exp_a = jnp.minimum(exp_a, 100.0)
            actor_loss = -masked_mean(exp_a * log_prob, mask)
            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': masked_mean(adv, mask),
                'bc_log_prob': masked_mean(log_prob, mask),
            }

        # DDPG+BC.
        q_actions = jnp.clip(dist.mode() if self.config['const_std'] else dist.sample(seed=rng), -1, 1)
        q = self.qhat(batch, q_actions)
        if self.config['mu_ensemble_size'] > 1:
            # Lower confidence bound over the ensemble. sqrt(var + eps) keeps the gradient finite when members agree.
            q = q.mean(0) - self.config['disagreement_coef'] * jnp.sqrt(q.var(0) + 1e-8)
        q_mask = jnp.ones_like(mask) if self.config['actor_q_unmasked'] else mask
        q_loss = -masked_mean(q, q_mask) / jax.lax.stop_gradient(jnp.abs(masked_mean(q, q_mask)) + 1e-6)
        bc_loss = -self.config['alpha'] * masked_mean(log_prob, mask)
        actor_loss = q_loss + bc_loss
        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': masked_mean(q, q_mask),
            'bc_log_prob': masked_mean(log_prob, mask),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        model_loss = 0.0
        if self.config['grounding'] == 'scalar_q':
            model_loss, model_info = self.q_reg_loss(batch, grad_params)
            for k, v in model_info.items():
                info[f'q/{k}'] = v
        elif self.config['actor_loss'] == 'ddpgbc':
            model_loss, model_info = self.mu_loss(batch, grad_params)
            for k, v in model_info.items():
                info[f'mu/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return value_loss + model_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.mode() if self.config['const_std'] else dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert config['grounding'] in ('mu', 'scalar_q')
        assert config['actor_loss'] in ('ddpgbc', 'awr', 'q_awr')
        assert config['actor_loss'] != 'q_awr' or config['grounding'] == 'scalar_q', "q_awr needs grounding='scalar_q'"

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        num_models = config['mu_ensemble_size']

        value_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['rep_dim'],
            layer_norm=config['layer_norm'],
            ensemble=False,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        if config['grounding'] == 'scalar_q':
            model_name, out_dim = 'q', 1
            ex_model_input = jnp.concatenate([ex_observations, ex_goals, ex_actions], axis=-1)
        else:
            model_name, out_dim = 'mu', config['rep_dim']
            ex_model_input = jnp.concatenate([ex_observations, ex_actions], axis=-1)
        model_cls = ensemblize(MLP, num_models) if num_models > 1 else MLP
        model_def = model_cls(
            (*config['mu_hidden_dims'], out_dim), activate_final=False, layer_norm=config['layer_norm']
        )

        network_info = {
            'value': (value_def, (ex_observations, ex_goals)),
            'target_value': (copy.deepcopy(value_def), (ex_observations, ex_goals)),
            model_name: (model_def, (ex_model_input,)),
            'actor': (actor_def, (ex_observations, ex_goals)),
        }
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='afv',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            mu_hidden_dims=(512, 512),  # One-step model hidden dimensions.
            rep_dim=512,  # Dimension d of phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile for the value.
            clip_value_target=True,  # Clip value targets and Q-hat to the range implied by the reward.
            mu_ensemble_size=5,  # Number of one-step models.
            disagreement_coef=1.0,  # Weight of the ensemble standard deviation in the lower confidence bound.
            mu_residual=True,  # Predict mu(s, a) = phi(s) + Delta(s, a) instead of phi(s') directly.
            actor_q_reward=True,  # Include r(s, g) and m(s, g) at the actor goal in Q-hat.
            actor_loss='ddpgbc',  # Actor loss type ('ddpgbc', 'awr', or 'q_awr').
            alpha=0.3,  # BC coefficient in DDPG+BC.
            awr_temperature=3.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # Ablations.
            grounding='mu',  # 'mu' (factorized one-step model) or 'scalar_q' (one q(s, g, a) head).
            q_expectile=0.5,  # Expectile of the q head regression (grounding='scalar_q').
            actor_q_unmasked=False,  # Apply the Q term of DDPG+BC on all rows (BC stays on labeled rows).
            mu_full_labels=False,  # Train mu on all rows (label oracle).
            actor_full_labels=False,  # Train the actor on all rows (label oracle).
            # Dataset hyperparameters.
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

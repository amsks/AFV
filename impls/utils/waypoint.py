"""Evaluation-time waypoints over a bank of dataset states, for agents with a quasimetric value (QRL).

For a quasimetric d whose triangle inequality is tight along optimal paths, a policy trained only on near goals
can reach far goals by following waypoints:

    w*(s, g) = argmin_{w in bank} [ d(s, w) + d(w, g) ]   s.t.  d(s, w) in [band_lo * k, band_hi * k],

where k is the goal horizon the actor was trained on (`actor_offset_max`). The bank is a uniform subsample of
dataset states, so d is only queried on the data manifold. With `multihop_radius > 0`, d(w, g) is replaced by the
shortest-path cost to the goal over the bank graph whose edges are pairs closer than `multihop_radius`; this
avoids relying on d at long range, where it saturates.

Nothing here reads an action. `bank_size=0` leaves evaluation unchanged.
"""

import heapq

import jax
import jax.numpy as jnp
import numpy as np


def bank_from_dataset(dataset, size, seed=0):
    """Uniformly subsample `size` observations from the dataset."""
    obs = np.asarray(dataset['observations'])
    idxs = np.random.RandomState(seed).choice(obs.shape[0], size=min(size, obs.shape[0]), replace=False)
    return np.array(obs[np.sort(idxs)], dtype=np.float32)


def select_waypoint(d_sw, d_wg, k_lo, k_hi):
    """Bank index minimizing d(s, w) + d(w, g) with d(s, w) in [k_lo, k_hi]. Returns (index, any_feasible)."""
    feasible = (d_sw >= k_lo) & (d_sw <= k_hi) & np.isfinite(d_wg)
    total = np.where(feasible, d_sw + d_wg, np.inf)
    return int(np.argmin(total)), bool(feasible.any())


def multihop_cost_to_goal(pair_d, d_wg, radius):
    """Shortest-path cost from every bank state to the goal, using only edges shorter than `radius` (Dijkstra)."""
    m = pair_d.shape[0]
    cost = np.where(d_wg <= radius, d_wg, np.inf)
    heap = [(c, i) for i, c in enumerate(cost) if np.isfinite(c)]
    heapq.heapify(heap)
    preds = [np.nonzero(pair_d[:, j] <= radius)[0] for j in range(m)]
    while heap:
        c, j = heapq.heappop(heap)
        if c > cost[j]:
            continue
        for i in preds[j]:
            nc = pair_d[i, j] + c
            if nc < cost[i]:
                cost[i] = nc
                heapq.heappush(heap, (nc, i))
    return cost


class WaypointPolicy:
    """Wraps a QRL agent's `sample_actions`, replacing far goals with waypoints from a state bank.

    Args:
        agent: Trained QRL agent.
        bank: (M, obs_dim) array of dataset states.
        k: Goal horizon the actor was trained on, in environment steps.
        band: (lo, hi) multipliers on k giving the feasible ring for d(s, w).
        reach: Switch to a new waypoint once d(s, w) falls to this.
        recompute_every: Re-plan at least this often.
        multihop_radius: If > 0, rank waypoints by the bank-graph cost to the goal instead of d(w, g).
    """

    def __init__(self, agent, bank, k, band=(0.5, 1.5), reach=5.0, recompute_every=25, multihop_radius=0.0):
        self.agent = agent
        self.bank = np.asarray(bank, dtype=np.float32)
        self.k_lo, self.k_hi = band[0] * k, band[1] * k
        self.reach = reach
        self.recompute_every = recompute_every
        self.multihop_radius = multihop_radius

        value = agent.network.select('value')
        embed = lambda x: value(x, x, info=True)[1]
        dist = lambda phi_a, phi_b: value(phi_a, phi_b, is_phi=True)
        self.bank_phi = embed(jnp.asarray(self.bank))
        self._embed = jax.jit(embed)
        self._dist = jax.jit(dist)
        self._dist_to_bank = jax.jit(lambda phi: dist(jnp.broadcast_to(phi, self.bank_phi.shape), self.bank_phi))
        self._dist_from_bank = jax.jit(lambda phi: dist(self.bank_phi, jnp.broadcast_to(phi, self.bank_phi.shape)))
        self._pair_d = None
        if multihop_radius > 0:
            self._pair_d = np.stack([np.asarray(self._dist_to_bank(self.bank_phi[i])) for i in range(len(self.bank))])
        self._d_wg_cache = {}
        self.reset()

    def reset(self):
        """Clear per-episode state. Called by `evaluate` at every episode start."""
        self._goal = None
        self._goal_phi = None
        self._d_wg = None
        self._idx = None
        self._steps_held = 0
        self._cooldown = 0

    def _new_goal(self, goal):
        self._goal = np.array(goal)
        self._goal_phi = self._embed(jnp.asarray(goal))
        key = self._goal.tobytes()
        if key not in self._d_wg_cache:
            d_wg = np.asarray(self._dist_from_bank(self._goal_phi))
            if self.multihop_radius > 0:
                d_wg = multihop_cost_to_goal(self._pair_d, d_wg, self.multihop_radius)
            self._d_wg_cache[key] = d_wg
        self._d_wg = self._d_wg_cache[key]
        self._idx = None
        self._steps_held = 0
        self._cooldown = 0

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        if self._goal is None or not np.array_equal(self._goal, goals):
            self._new_goal(goals)
        phi_s = self._embed(jnp.asarray(observations))
        target = goals
        if float(self._dist(phi_s, self._goal_phi)) > self.k_hi:
            d_cur = np.inf if self._idx is None else float(self._dist(phi_s, self.bank_phi[self._idx]))
            replan = self._idx is None or self._steps_held >= self.recompute_every or d_cur <= self.reach
            if replan and self._cooldown > 0:
                # No feasible waypoint was found recently; head for the goal for a while before trying again.
                self._cooldown -= 1
                replan = False
            if replan:
                idx, ok = select_waypoint(np.asarray(self._dist_to_bank(phi_s)), self._d_wg, self.k_lo, self.k_hi)
                if ok:
                    self._idx, self._steps_held = idx, 0
                else:
                    self._idx, self._cooldown = None, self.recompute_every
            if self._idx is not None:
                target = self.bank[self._idx]
                self._steps_held += 1
        else:
            # Within the actor's horizon: go straight for the goal.
            self._idx, self._steps_held, self._cooldown = None, 0, 0
        return self.agent.sample_actions(observations=observations, goals=target, seed=seed, temperature=temperature)

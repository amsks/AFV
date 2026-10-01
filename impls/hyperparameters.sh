# Commands for every result in the paper. Run from impls/. Each command is run with --seed=0, 1, 2.
#
# Common settings (all runs):
#   --train_steps=500000 --eval_interval=100000 --save_interval=100000 --eval_episodes=50
# Reported number: mean success rate over the last three evaluations (300k, 400k, 500k steps).
#
# Label masks: M=masks. A mask keeps the action labels of a fraction p of the trajectories (label_seed=0 unless
# stated otherwise). Generate them first:
#   python ../data_gen_scripts/make_label_masks.py --env_name=ENV --p=P --label_seed=SEED --save_path=$M/ENV_pP_sSEED.npz

# ===================================================================================================================
# Cube (cube-single-play-v0; cube-double-play-v0 at p=0.10)
# ===================================================================================================================

# AFV (ours): bilinear value from all rows, displacement one-step model and DDPG+BC on labeled rows.
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz
# AFV, one-step model predicting phi(s') directly (no displacement).
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.mu_residual=False
# p in {0.02, 0.05, 0.10, 0.20}: change the mask. Label draws 1 and 2: masks made with --label_seed=1, 2.

# SS-ORL (GCIVL): (1) inverse dynamics model on labeled rows, (2) GCIVL on the actions it fills in.
python main.py --env_name=cube-single-play-v0 --agent=agents/idm.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --train_steps=100000 --eval_interval=0 --run_group=idm_cube_p0.1
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl.py --agent.alpha=10.0 --impute_path=exp/idm_cube_p0.1/RUN/imputed_actions.npz

# GCIQL with every action-dependent loss (including the value loss) on labeled rows only.
python main.py --env_name=cube-single-play-v0 --agent=agents/gciql.py --agent.alpha=1.0 --agent.mask_value_loss=True --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz

# Full-label references.
python main.py --env_name=cube-single-play-v0 --agent=agents/gciql.py --agent.alpha=1.0
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl.py --agent.alpha=10.0

# ===================================================================================================================
# AntMaze (antmaze-medium-navigate-v0, antmaze-large-navigate-v0)
# ===================================================================================================================

# AFV-QRL: quasimetric value from all rows, latent dynamics model and DDPG+BC on labeled rows.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz

# AFV-GEO: actor trained only on goals at most 25 steps ahead (AWR), plus waypoints from 512 dataset states at evaluation.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.actor_offset_max=25 --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz --waypoint_bank_size=512 --waypoint_k=15
# antmaze-large: --waypoint_k=25 --waypoint_multihop_radius=30.

# SS-ORL (QRL), and SS-ORL + waypoints (same flags as AFV-GEO, with --impute_path instead of --label_mask_path).
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/idm.py --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz --train_steps=100000 --eval_interval=0 --run_group=idm_ammed_p0.1
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --impute_path=exp/idm_ammed_p0.1/RUN/imputed_actions.npz
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --agent.actor_loss=awr --agent.alpha=3.0 --agent.actor_offset_max=25 --impute_path=exp/idm_ammed_p0.1/RUN/imputed_actions.npz --waypoint_bank_size=512 --waypoint_k=15

# HIQL with the low-level actor on labeled rows only.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz

# Full-label reference.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py

# ===================================================================================================================
# Ablations (cube-single-play-v0, p=0.10 unless stated otherwise)
# ===================================================================================================================

# Q term of DDPG+BC applied on all rows instead of labeled rows only.
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.actor_q_unmasked=True
# One q(s, g, a) head instead of the factorized one-step model: DDPG+BC (first) and AWR at p=0.10 and p=0.05 (second).
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.grounding=scalar_q
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.grounding=scalar_q --agent.actor_loss=q_awr
# AWR with the value advantage, no one-step model (p in {0.02, 0.05, 0.10}).
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.actor_loss=awr --agent.mu_ensemble_size=1
# Corrupted labels: shuffled among labeled rows, or replaced with uniform random actions.
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --corrupt_labels=shuffle
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --corrupt_labels=uniform
# Label oracles: actor or one-step model trained on all rows.
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.actor_full_labels=True
python main.py --env_name=cube-single-play-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.1_s0.npz --agent.mu_full_labels=True
# AntMaze label oracles (QRL): actor or dynamics model trained on all rows.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz --agent.actor_full_labels=True
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz --agent.dynamics_full_labels=True
# QRL dynamics model predicting the next latent directly (no residual), at full labels and at p=0.10.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --agent.dynamics_residual=False
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --agent.dynamics_residual=False --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz
# AFV (bilinear value) on antmaze-medium, with and without the displacement one-step model.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/afv.py --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz --agent.mu_residual=False
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/afv.py --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz
# Waypoints at evaluation only, on an actor trained on all goals (re-score a saved checkpoint).
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --eval_only=1 --restore_path=exp/RUN_GROUP/RUN --restore_epoch=500000 --waypoint_bank_size=512 --waypoint_k=50
# DDPG+BC instead of AWR in AFV-GEO.
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/qrl.py --agent.actor_offset_max=25 --label_mask_path=$M/antmaze-medium-navigate-v0_p0.1_s0.npz --waypoint_bank_size=512 --waypoint_k=15
# Observation noise: build noisy datasets (python ../data_gen_scripts/add_obs_noise.py --dataset=cube-single-play-v0),
# then run AFV and SS-ORL (GCIVL) at p=0.05 with --env_name=cube-single-sig{0p1,0p3,1p0}-v0.
python main.py --env_name=cube-single-sig0p1-v0 --agent=agents/afv.py --label_mask_path=$M/cube-single-play-v0_p0.05_s0.npz

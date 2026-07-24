# World Models & Real-Time Decision Systems Curriculum

### Understanding RL, learned dynamics, latent planning, and autonomous decision loops

*The principle: reproduce the field's canonical methods at small scale before trying to build a real-time autonomous system. Read the paper -> rebuild the core from scratch -> verify against a baseline -> write down what the model can predict, what it cannot predict, and how that affects decisions.*

The north-star loop:

```text
streaming observation
  -> state / belief update
  -> world model prediction
  -> planning or policy decision
  -> action
  -> feedback
  -> model / policy update
```

---

## How to use this document

**Calibration.** This assumes your current `ml-journey` base: ML fundamentals, micrograd, makemore, early GPT/self-attention work, LoRA/quantization/distillation, and continual-learning experiments on Split-MNIST. You do **not** need to restart from basic neural networks. The missing layer is online decision-making: environments, rewards, transitions, rollouts, policies, planning, uncertainty, and real-time evaluation.

**Hardware baseline.**
- CPU is enough for bandits, Gridworld, CartPole, tabular RL, low-dimensional dynamics, and small MiniGrid experiments.
- Google Colab or Kaggle T4 is enough for MiniGrid image observations, `CarRacing-v3`, small RSSM/Dreamer-style experiments, and trajectory transformers.
- Use MuJoCo only after the small loop works. Use Isaac Lab only when you are ready for robotics-style simulation and GPU-heavy parallel environments.

**Core stack.** `torch`, `torchvision`, `gymnasium`, `minigrid`, `stable-baselines3`, `cleanrl`, `numpy`, `matplotlib`, `tqdm`, `wandb`. Add `mujoco` later. Keep the first reproductions small enough that you can rerun them often.

**Default environments** (small first = fast loop = more experiments):
- First control loops: `CartPole-v1`, `MountainCar-v0`
- Discrete planning: MiniGrid `Empty`, `FourRooms`, `DoorKey`
- Visual control: `CarRacing-v3`
- Later continuous control: MuJoCo `HalfCheetah`, `Walker2d`, `Hopper`

**The loop for every practical.** (1) Read the paper or docs. Skim for the problem, then go deep on the method and loss functions. (2) Reproduce the minimal core from scratch. Do not use a high-level wrapper for the part you are trying to learn. (3) Verify against a known baseline, a reference implementation, or a simpler agent. (4) Commit a folder with a README stating the result. (5) Do one probe or ablation and write one paragraph: *what did the model fail to predict, and why does that matter for decisions?*

**Deliverable convention.** Each practical lives under `research/world_models/` with a runnable notebook or script, a README, curves or rollout videos where relevant, and the metric that proves it worked. Use this README shape:

```text
problem -> method -> result -> latency -> failure mode -> what I learned
```

**Reading notes convention.** For every required paper or blog, create a short note under:

```text
research/world_models/reading_notes/
  000_rl_foundations.md
  001_model_free_baselines.md
  002_classic_model_based_rl.md
  003_world_models_planet.md
  004_dreamer_family.md
  005_muzero_value_equivalent_models.md
  006_transformer_foundation_world_models.md
  007_realtime_control_and_robotics.md
```

Each note should contain: one-sentence thesis, algorithm sketch, key losses/equations, what to implement, what to measure, one failure mode, and how it connects to real-time decision-making.

**Source verification.** The links in this curriculum were checked against primary sources, official docs, project pages, or reference repos on 2026-07-08.

---

# Pillar 0 - Inputs: online decision-making foundations

## WM0.1 - Agent-environment loop

**Goal.** Stop thinking only in static datasets. Build the online loop where an agent repeatedly observes, acts, receives feedback, and updates.

**Read.** Sutton & Barto, *Reinforcement Learning: An Introduction*, chapters 1-3 - https://incompleteideas.net/book/the-book-2nd.html - Gymnasium basic usage - https://gymnasium.farama.org/introduction/basic_usage/ - OpenAI Spinning Up, key RL concepts - https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

**Specs.** Pure Python + `gymnasium`. `CartPole-v1`, then MiniGrid `Empty`. CPU. Effort: 0.5-1 day.

**How.** (1) Create a reusable `Agent` interface with `observe`, `act`, and `update`. (2) Run random, heuristic, and simple table-based agents. (3) Log episode return, episode length, action distribution, and failure states. (4) Add a strict per-action time budget, even if the agent is simple.

**Deliverable & success.** `wm0_agent_loop.ipynb`; you can explain observation, action, reward, done/truncated, transition, return, policy, episode, and why an autonomous system needs a latency budget.

## WM0.2 - Bandits and exploration

**Goal.** Understand exploration vs exploitation before adding deep networks.

**Read.** Sutton & Barto, chapter 2 - https://incompleteideas.net/book/the-book-2nd.html - Lilian Weng, *A Long Peek into Reinforcement Learning* - https://lilianweng.github.io/posts/2018-02-19-rl-overview/

**Specs.** Pure Python. Stationary and non-stationary bandits. CPU. Effort: 0.5 day.

**How.** (1) Implement epsilon-greedy. (2) Implement UCB. (3) Implement Thompson sampling. (4) Compare cumulative regret across changing reward distributions.

**Deliverable & success.** `wm0_bandits.ipynb`; a regret plot showing why real-time decision models need exploration, uncertainty, or fallback behavior.

## WM0.3 - Tabular MDPs

**Goal.** Own value functions and Bellman backups before deep RL.

**Read.** Sutton & Barto, chapters 3-6 - http://incompleteideas.net/book/the-book-2nd.html - OpenAI Spinning Up, model-free vs model-based taxonomy - https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html

**Specs.** Small Gridworld. Deterministic and stochastic variants. CPU. Effort: 1 day.

**How.** (1) Implement policy evaluation. (2) Implement value iteration. (3) Implement Q-learning. (4) Implement SARSA. (5) Compare learned policies on deterministic and stochastic Gridworlds.

**Deliverable & success.** `wm0_gridworld.ipynb`; you can explain value, Q-value, Bellman update, bootstrapping, discount factor, and why stochastic transitions change the optimal policy.

---

# Pillar 1 - Model-free RL baselines

## WM1.1 - DQN from scratch

**Goal.** Build the simplest deep RL baseline that learns from replay, so later world-model agents have something honest to beat.

**Read.** Mnih et al., *Playing Atari with Deep Reinforcement Learning* - https://arxiv.org/abs/1312.5602 - CleanRL DQN docs - https://docs.cleanrl.dev/rl-algorithms/dqn/ - Stable-Baselines3 DQN docs - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

**Specs.** `CartPole-v1`, then `LunarLander-v3`. CPU or T4. Effort: 1-2 days.

**How.** (1) Implement replay buffer. (2) Implement Q-network. (3) Implement target network. (4) Implement epsilon decay. (5) Track return, Q-values, TD error, replay size, update frequency, and inference latency.

**Deliverable & success.** `wm1_dqn_cartpole.ipynb`; DQN solves `CartPole-v1` across at least 3 seeds and produces a learning curve.

## WM1.2 - Policy gradients and actor-critic

**Goal.** Learn policy optimization before PPO.

**Read.** OpenAI Spinning Up policy-gradient material - https://spinningup.openai.com - Hugging Face Deep RL Course, Unit 1 - https://huggingface.co/learn/deep-rl-course/en/unit1/introduction

**Specs.** `CartPole-v1`, `LunarLander-v3`. CPU/T4. Effort: 1-2 days.

**How.** (1) Implement REINFORCE. (2) Add a learned value baseline. (3) Implement advantage actor-critic. (4) Compare variance and learning stability.

**Deliverable & success.** `wm1_policy_gradient_actor_critic.ipynb`; you can explain why policy-gradient estimates are noisy and why a value baseline helps.

## WM1.3 - PPO baseline

**Goal.** Get a reliable model-free baseline for later world-model comparisons.

**Read.** Schulman et al., *Proximal Policy Optimization Algorithms* - https://arxiv.org/abs/1707.06347 - Stable-Baselines3 PPO docs - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html - CleanRL docs - https://docs.cleanrl.dev

**Specs.** MiniGrid `Empty`, MiniGrid `FourRooms`, `CarRacing-v3`. T4 recommended for visual tasks. Effort: 1-2 days.

**How.** (1) Train PPO with Stable-Baselines3. (2) Re-run with at least 3 seeds. (3) Record sample efficiency, final return, wall-clock time, and inference latency. (4) Compare against a CleanRL implementation at a high level.

**Deliverable & success.** `wm1_ppo_baselines.ipynb`; you have model-free baseline numbers every world-model agent must beat or explain.

---

# Pillar 2 - First world model from scratch

## WM2.1 - Dynamics model on low-dimensional state

**Goal.** Learn the basic world-model prediction task without pixels.

**Read.** Ha & Schmidhuber, *World Models* - https://arxiv.org/abs/1803.10122 - World Models interactive article - https://worldmodels.github.io

**Specs.** `CartPole-v1` state vectors. CPU. Effort: 1 day.

**How.** (1) Collect random and policy-generated trajectories. (2) Train `state_t + action_t -> state_t+1`. (3) Add reward prediction. (4) Add done prediction. (5) Evaluate 1-step, 5-step, and 20-step rollout error.

**Deliverable & success.** `wm2_low_dim_dynamics.ipynb`; your model predicts short-horizon dynamics better than a naive baseline, and you can show where multi-step predictions drift.

## WM2.2 - Planning with a learned dynamics model

**Goal.** Use the learned model to choose actions, not just predict.

**Read.** Sutton, *Dyna, an Integrated Architecture for Learning, Planning, and Reacting* - https://dl.acm.org/doi/10.1145/122344.122377 - OpenAI Spinning Up, key papers in deep RL - https://spinningup.openai.com/en/latest/spinningup/keypapers.html

**Specs.** `CartPole-v1` or simple Gridworld. CPU. Effort: 1 day.

**How.** (1) Implement random shooting: sample action sequences, predict future rewards, choose the first action from the best sequence. (2) Implement model predictive control: re-plan after every real observation. (3) Compare random policy, DQN/PPO, and model-based planner. (4) Add a per-action planning time budget.

**Deliverable & success.** `wm2_random_shooting_mpc.ipynb`; the planner improves over random and exposes the tradeoff between planning quality and latency.

## WM2.3 - Pixel encoder and latent dynamics

**Goal.** Move from raw observations to compressed latent state.

**Read.** Ha & Schmidhuber, *World Models* - https://arxiv.org/abs/1803.10122 - Gymnasium docs - https://gymnasium.farama.org - MiniGrid docs - https://minigrid.farama.org

**Specs.** MiniGrid image observations first; `CarRacing-v3` second. T4. Effort: 2-3 days.

**How.** (1) Train a convolutional autoencoder or VAE: `frame -> latent z -> reconstructed frame`. (2) Train latent dynamics: `z_t + action_t -> z_t+1`. (3) Add reward and done heads. (4) Evaluate reconstruction quality, latent prediction loss, and multi-step latent drift.

**Deliverable & success.** `wm2_pixel_encoder_latent_dynamics.ipynb`; you can compress observations and predict useful future latent states.

---

# Pillar 3 - Latent planning

## WM3.1 - PlaNet-style latent planner

**Goal.** Plan in latent space instead of pixel space.

**Read.** Hafner et al., *Learning Latent Dynamics for Planning from Pixels* - https://arxiv.org/abs/1811.04551 - PlaNet project page - https://danijar.com/project/planet/ - PlaNet official code - https://github.com/google-research/planet

**Specs.** MiniGrid or `CarRacing-v3`. T4. Effort: 3-5 days.

**How.** (1) Build a recurrent latent dynamics model with deterministic hidden state and stochastic latent state. (2) Train on trajectory batches with reconstruction, reward, and latent losses. (3) Implement CEM planning over action sequences. (4) Execute only the first planned action, observe, and re-plan.

**Deliverable & success.** `wm3_latent_planner.ipynb`; your agent uses latent imagined futures to act in the real environment.

## WM3.2 - Planning diagnostics

**Goal.** Learn why learned-model planners fail.

**Read.** Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (PETS) - https://arxiv.org/abs/1805.12114 - Janner et al., *When to Trust Your Model: Model-Based Policy Optimization* - https://arxiv.org/abs/1906.08253 - BAIR MBPO blog - https://bair.berkeley.edu/blog/2019/12/12/mbpo/

**Specs.** Reuse WM2.2 or WM3.1. CPU/T4 depending on environment. Effort: 1-2 days.

**How.** (1) Compare real rollouts vs imagined rollouts. (2) Track model error by horizon. (3) Track reward prediction error. (4) Check out-of-distribution actions selected by the planner. (5) Add uncertainty or action penalties to reduce model exploitation.

**Deliverable & success.** `wm3_planning_diagnostics.ipynb`; a report showing at least three failure modes: compounding error, reward misprediction, and planner exploiting model errors.

## WM3.3 - Short-horizon model-based policy optimization

**Goal.** Understand why short imagined rollouts are often safer than long ones.

**Read.** Feinberg et al., *Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning* - https://arxiv.org/abs/1803.00101 - Janner et al., *When to Trust Your Model: Model-Based Policy Optimization* - https://arxiv.org/abs/1906.08253

**Specs.** Low-dimensional continuous or discrete control. CPU/T4. Effort: 2-3 days.

**How.** (1) Train a dynamics model from real replay. (2) Generate short synthetic rollouts. (3) Mix synthetic and real transitions into a model-free learner. (4) Sweep rollout horizon and watch when model bias hurts.

**Deliverable & success.** `wm3_short_horizon_mbpo.ipynb`; a horizon-vs-return plot showing the model-bias tradeoff.

---

# Pillar 4 - Dreamer-style imagination learning

## WM4.1 - Recurrent state-space model

**Goal.** Build the model core used by Dreamer-style agents.

**Read.** Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* - https://arxiv.org/abs/1912.01603 - Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3) - https://arxiv.org/abs/2301.04104 - Dreamer project page - https://danijar.com/project/dreamer/

**Specs.** MiniGrid first; `CarRacing-v3` later. T4. Effort: 3-5 days.

**How.** (1) Implement RSSM: deterministic recurrent state, stochastic latent state, posterior from current observation, prior from previous state plus action. (2) Train reconstruction, reward, continuation, and KL losses. (3) Log posterior/prior KL, reconstruction loss, reward loss, and continuation accuracy.

**Deliverable & success.** `wm4_rssm.ipynb`; a trained world model that supports imagined latent rollouts.

## WM4.2 - Actor-critic from imagination

**Goal.** Train behavior inside the world model.

**Read.** Hafner et al., *Dream to Control* - https://arxiv.org/abs/1912.01603 - DreamerV3 project page - https://danijar.com/project/dreamerv3/ - DreamerV3 code - https://github.com/danijar/dreamerv3

**Specs.** Reuse WM4.1 world model. MiniGrid or `CarRacing-v3`. T4. Effort: 3-5 days.

**How.** (1) Freeze or jointly update the world model. (2) Roll out imagined trajectories in latent space. (3) Train actor to maximize imagined returns. (4) Train critic to predict imagined values. (5) Execute actor in the real environment and collect more data.

**Deliverable & success.** `wm4_imagination_actor_critic.ipynb`; the agent improves using imagined rollouts, not just real environment transitions.

## WM4.3 - MiniDreamer

**Goal.** Create your first serious world-model portfolio project.

**Read.** Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2) - https://arxiv.org/abs/2010.02193 - Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3) - https://arxiv.org/abs/2301.04104

**Specs.** MiniGrid or `CarRacing-v3`. T4. Effort: 1-2 weeks.

**How.** (1) Package the world model, actor-critic, evaluator, configs, and notebooks. (2) Run PPO as a model-free baseline. (3) Train MiniDreamer with fixed seeds. (4) Report score, environment steps, wall-clock time, inference latency, reward prediction error, horizon drift, and rollout video.

**Deliverable & success.** `research/world_models/minidreamer/` with `README.md`, `train_world_model.py`, `train_actor_critic.py`, `evaluate.py`, `configs/`, and `notebooks/`; a clean comparison against PPO.

## WM4.4 - Frontier Dreamer reading

**Goal.** Understand where scalable world-model agents are heading without trying to reproduce the full frontier system yet.

**Read.** Hafner, Yan, Lillicrap, *Training Agents Inside of Scalable World Models* (Dreamer 4) - https://arxiv.org/abs/2509.24527 - Dreamer 4 project page - https://danijar.com/project/dreamer4/

**Specs.** Reading and design note only. No implementation required yet. Effort: 0.5 day.

**How.** (1) Compare Dreamer, DreamerV2, DreamerV3, and Dreamer 4. (2) Identify which changes are algorithmic vs scaling/engineering. (3) Write what is realistic to borrow for your small MiniDreamer.

**Deliverable & success.** `004_dreamer_family.md`; you can explain RSSM, posterior/prior latents, KL balancing, reward and continuation heads, imagined rollouts, actor-critic from imagination, and what changed across the Dreamer line.

---

# Pillar 5 - Value-equivalent world models

## WM5.1 - MuZero concepts

**Goal.** Understand the branch that predicts what matters for planning rather than reconstructing the whole world.

**Read.** Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* - https://arxiv.org/abs/1911.08265 - Google DeepMind MuZero overview - https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/

**Specs.** Tic-Tac-Toe, Connect Four, or small Gridworld. CPU. Effort: 2-4 days.

**How.** (1) Implement representation: `observation -> hidden state`. (2) Implement dynamics: `hidden state + action -> next hidden state + reward`. (3) Implement prediction: `hidden state -> policy + value`. (4) Train the model without requiring pixel reconstruction.

**Deliverable & success.** `wm5_tiny_muzero.ipynb`; you can explain why MuZero predicts reward, value, and policy rather than reconstructing observations.

## WM5.2 - Search with a learned model

**Goal.** Combine learned dynamics with tree search.

**Read.** MuZero paper again, focusing on MCTS and target construction - https://arxiv.org/abs/1911.08265

**Specs.** Same small game as WM5.1. CPU. Effort: 2-3 days.

**How.** (1) Implement MCTS on a known small game. (2) Replace true dynamics with learned dynamics. (3) Compare planning using true simulator vs learned simulator. (4) Track value error and search depth.

**Deliverable & success.** `wm5_learned_model_mcts.ipynb`; you can explain when model fidelity matters and when value-equivalent prediction is enough.

---

# Pillar 6 - Transformer and foundation world models

## WM6.1 - Dynamics as sequence modeling

**Goal.** Connect your GPT work to world models.

**Read.** Micheli, Alonso, Fleuret, *Transformers are Sample-Efficient World Models* (IRIS) - https://arxiv.org/abs/2209.00588 - IRIS code - https://github.com/eloialonso/iris

**Specs.** MiniGrid trajectories. CPU/T4. Effort: 2-4 days.

**How.** (1) Convert trajectories into token sequences: `obs_t, action_t, reward_t, obs_t+1, ...`. (2) Train a small transformer to predict next observation token and reward. (3) Use it for short imagined rollouts. (4) Compare against a non-transformer dynamics model.

**Deliverable & success.** `wm6_trajectory_transformer.ipynb`; a transformer dynamics model that predicts future tokens and rewards from trajectory history.

## WM6.2 - Discrete visual token world model

**Goal.** Move toward IRIS-style visual world modeling.

**Read.** IRIS paper and code again - https://arxiv.org/abs/2209.00588 - https://github.com/eloialonso/iris

**Specs.** MiniGrid image observations first; `CarRacing-v3` optional. T4. Effort: 3-5 days.

**How.** (1) Train or use a discrete autoencoder for frames. (2) Turn frames into discrete latent tokens. (3) Train a transformer over tokenized trajectory sequences. (4) Evaluate imagined rollout quality and reward prediction.

**Deliverable & success.** `wm6_discrete_visual_tokens.ipynb`; a small IRIS-inspired world model that treats visual dynamics like language modeling.

## WM6.3 - Foundation world model reading

**Goal.** Understand the difference between a small control world model and a general interactive environment model.

**Read.** Bruce et al., *Genie: Generative Interactive Environments* - https://arxiv.org/abs/2402.15391 - Genie project page - https://sites.google.com/view/genie-2024/home - Google DeepMind Genie model page - https://deepmind.google/models/genie/ - V-JEPA 2, *Self-Supervised Video Models Enable Understanding, Prediction and Planning* - https://arxiv.org/html/2506.09985v1 - V-JEPA 2 Meta AI page - https://ai.meta.com/research/vjepa/

**Specs.** Reading and design note first. Small reproduction optional later. Effort: 1 day.

**How.** (1) Explain observation tokenization, autoregressive dynamics, learned latent action spaces, and representation-space prediction. (2) Compare Genie-style interactive video models with Dreamer-style control models. (3) Identify which ideas can realistically transfer into your small real-time agent.

**Deliverable & success.** `006_transformer_foundation_world_models.md`; you can explain why a foundation world model is not the same thing as a small control world model.

---

# Pillar 7 - Real-time control and autonomous systems

## WM7.1 - Latency-aware agent runtime

**Goal.** Build the runtime shape for real-time decision systems.

**Read.** Gymnasium docs - https://gymnasium.farama.org - MiniGrid docs - https://minigrid.farama.org - MuJoCo docs - https://mujoco.readthedocs.io

**Specs.** Start with MiniGrid or CartPole. Add MuJoCo only after the runtime works. CPU/T4. Effort: 2-3 days.

**How.** (1) Create a streaming observation interface. (2) Maintain a recurrent state or belief state. (3) Add a planner or policy with a fixed time budget. (4) Add fallback behavior when the model is uncertain or late. (5) Log observation summary, latent state, candidate actions, predicted reward/risk, chosen action, latency, and real outcome.

**Deliverable & success.** `wm7_latency_aware_runtime/`; the agent continuously acts under a latency budget and produces an auditable decision trace.

## WM7.2 - Uncertainty and safety

**Goal.** Make the model know when not to trust itself.

**Read.** PETS - https://arxiv.org/abs/1805.12114 - MBPO - https://arxiv.org/abs/1906.08253 - BAIR MBPO blog - https://bair.berkeley.edu/blog/2019/12/12/mbpo/

**Specs.** Reuse WM7.1 runtime. CPU/T4. Effort: 2-4 days.

**How.** (1) Train an ensemble of dynamics models. (2) Measure disagreement across predicted futures. (3) Penalize high-uncertainty action sequences. (4) Add rule-based action constraints. (5) Add a safe fallback policy.

**Deliverable & success.** `wm7_uncertainty_safe_planner.ipynb`; the agent avoids actions where the world model is uncertain or outside its training distribution.

## WM7.3 - Continuous control bridge

**Goal.** Move from toy control to robotics-style simulation.

**Read.** MuJoCo site - https://mujoco.org - Gymnasium MuJoCo environments - https://gymnasium.farama.org/environments/mujoco/ - Isaac Lab overview - https://isaac-sim.github.io/IsaacLab/ - Isaac Lab RL docs - https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/index.html

**Specs.** MuJoCo `HalfCheetah`, `Walker2d`, or `Hopper`. T4 or stronger GPU if needed. Effort: 3-7 days.

**How.** (1) Train PPO or SAC baseline. (2) Measure simulator throughput, policy latency, and action bounds. (3) Train a simple dynamics model. (4) Compare model-based planning, policy-only control, and hybrid fallback. (5) Note contact dynamics and sim-to-real limitations.

**Deliverable & success.** `wm7_mujoco_control_bridge.ipynb`; a table for return, environment steps, wall-clock time, inference latency, and failure modes.

## WM7.4 - TD-MPC2 reading and reproduction sketch

**Goal.** Understand a modern real-time model-predictive control direction without overcommitting too early.

**Read.** TD-MPC2 paper - https://arxiv.org/abs/2310.16828 - TD-MPC2 project page - https://www.tdmpc2.com/ - TD-MPC2 code - https://github.com/nicklashansen/tdmpc2

**Specs.** Reading plus small design note first. Optional reproduction after WM7.3. Effort: 1 day reading; 1-2 weeks if reproducing.

**How.** (1) Identify the learned latent dynamics, value learning, and MPC pieces. (2) Compare against PlaNet, Dreamer, and MuZero. (3) Decide what part is worth reproducing at small scale.

**Deliverable & success.** `007_realtime_control_and_robotics.md`; you can explain why continuous control adds action bounds, contact dynamics, simulator speed, policy latency, and safety constraints.

---

# Synthesis - the project

Only after Pillars 0-4 do you build the first serious version of the thing this is for: a **real-time world-model decision engine**.

It should have:
- a streaming observation loop,
- a belief/state updater,
- a learned dynamics model,
- reward/risk prediction,
- short-horizon planning or an imagination-trained policy,
- uncertainty estimates,
- a fallback policy,
- latency tracking,
- decision logs,
- periodic model/policy updates from feedback.

At that point you will not be guessing. You will be composing methods you have reproduced: tabular RL, DQN/PPO baselines, learned dynamics, latent planning, Dreamer-style imagination, MuZero-style value-equivalent prediction, and transformer trajectory modeling.

---

# Suggested sequencing

- **Run this alongside `ml-journey`, do not replace it.** Your existing transformer and continual-learning work helps here; this plan adds the decision-making and world-modeling layer.
- **Order:** WM0.1-WM0.3 -> WM1.1-WM1.3 -> WM2.1-WM2.2 -> WM3.1-WM3.2 -> WM4.1-WM4.3. Treat WM5-WM7 as the advanced set once the first MiniDreamer-style project exists.
- **Pace honestly.** Small items are 0.5-2 days. Latent planning and Dreamer-style items are 3-7 days. MiniDreamer is 1-2 weeks. Do not move on until the metric exists.
- **Track everything.** Use `wandb` for curves and commit the README after each reproduction. If you cannot show return, prediction loss, drift, and latency, the experiment is not finished.

---

# First 30 days

## Week 1 - Online RL foundations

- WM0.1 agent-environment loop
- WM0.2 bandits
- WM0.3 tabular Gridworld
- Write `research/world_models/README.md`
- Write `000_rl_foundations.md`

**Checkpoint.** You can explain the RL loop and implement tabular learning without looking anything up.

## Week 2 - Deep RL baselines

- WM1.1 DQN from scratch on CartPole
- WM1.2 REINFORCE / actor-critic
- WM1.3 PPO with Stable-Baselines3
- Write `001_model_free_baselines.md`

**Checkpoint.** You have baseline curves and know how much data model-free RL needs.

## Week 3 - First world model

- WM2.1 low-dimensional dynamics model
- WM2.2 model-based planning
- Compare against DQN/PPO
- Write `002_classic_model_based_rl.md`

**Checkpoint.** Your learned model can plan short-horizon actions in a simple environment.

## Week 4 - Latent world model

- WM2.3 pixel encoder + latent dynamics
- WM3.1 PlaNet-style planner skeleton
- Reward/done prediction
- Rollout drift diagnostics
- Write `003_world_models_planet.md`

**Checkpoint.** You can show what the model predicts correctly, where it drifts, and whether it is useful for decisions.

---

# Global resources

- Sutton & Barto, *Reinforcement Learning: An Introduction* - https://incompleteideas.net/book/the-book-2nd.html
- Gymnasium docs - https://gymnasium.farama.org
- MiniGrid docs - https://minigrid.farama.org
- OpenAI Spinning Up - https://spinningup.openai.com
- Hugging Face Deep RL Course - https://huggingface.co/learn/deep-rl-course
- CleanRL - https://docs.cleanrl.dev
- Stable-Baselines3 - https://stable-baselines3.readthedocs.io
- World Models - https://worldmodels.github.io
- PlaNet - https://danijar.com/project/planet/
- DreamerV3 - https://danijar.com/project/dreamerv3/
- MuZero overview - https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
- IRIS - https://github.com/eloialonso/iris
- MuJoCo - https://mujoco.org
- Isaac Lab - https://isaac-sim.github.io/IsaacLab/
- TD-MPC2 - https://www.tdmpc2.com/

---

*Reproduce, verify, measure, write down what surprised you. Do that across the pillars and you will not just know world models as papers; you will know them as working decision systems with measurable limits.*

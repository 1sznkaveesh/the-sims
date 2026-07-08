# RL for quadruped locomotion — a concepts primer

*A reference for training quadrupeds (like the Stanford Pupper) using RSL_RL, mjlab, and MuJoCo. This is not a tutorial or a task list — it's a map of the concepts and vocabulary you'll run into, so nothing in the actual code feels unfamiliar. Read it once end-to-end, then keep it nearby while you work.*

You already know PPO, actor-critic, and policy gradients from the HuggingFace course. Nothing here replaces that — it shows you how those ideas get *applied* to a robot with 12 joints, running thousands of copies in parallel, on a GPU.

---

## 1. Who does what: the five pieces of the puzzle

Before anything else, get clear on which library does which job. Most of the early confusion in this field comes from treating "MuJoCo / PyTorch / RSL_RL" as one blurry stack instead of five components with sharply separated responsibilities. For each one, the most useful thing to know is what it has **no concept of** — that's what tells you where its job ends and the next one begins.

| Component | What it is | Its one job | Has no concept of |
|---|---|---|---|
| **MuJoCo** | A physics engine (C/C++) | Given a robot body + torques, compute the next physical state — contacts, gravity, joint dynamics | Rewards, actions in the RL sense, neural networks, PyTorch |
| **MuJoCo Warp** | A GPU backend for MuJoCo, built on NVIDIA Warp | Run *thousands of copies* of that same physics computation at once on a GPU, instead of one at a time on a CPU | Still nothing about RL — it's the same physics, just parallelized |
| **mjlab** | A Python framework built on MuJoCo Warp | Wrap raw parallel physics into an RL-shaped interface: define observations, actions, rewards, terminations, curriculum — and hand it all back and forth as **PyTorch tensors** | How to *learn* — it never runs a gradient update. It's also just one of several interchangeable env stacks (IsaacLab, Genesis, and PyBullet play the same role for other simulators) |
| **PyTorch** | A general-purpose tensor + autograd library | Store numbers as GPU tensors, do fast batched math, compute gradients automatically | Robots, physics, or RL at all. It's the shared language mjlab and RSL_RL both speak — not a participant in the RL problem itself |
| **RSL_RL** | The learning algorithm library (built by ETH Zurich) | Take whatever vectorized environment it's handed and run PPO — collect rollouts, update the actor-critic networks, log, checkpoint | **What a robot is.** It has never seen a joint or a leg — only tensors of numbers in, tensors of numbers out |

**The one sentence that ties it together:** MuJoCo simulates a body. mjlab turns that body into a well-defined problem. PyTorch is the shared language of numbers everything is written in. RSL_RL solves the problem — and would just as happily solve a completely different one (a robotic arm, a drone) if handed a different environment with the same interface.

---

## 2. The three-layer architecture

Everything in this document sits inside one of three layers. Whenever you're unsure "which part of the system am I even looking at," come back to this diagram.

```
┌─────────────────────────────────────────────────────┐
│ LAYER 3 — LEARNING ALGORITHM (RSL_RL)                │
│ PPO, actor-critic networks, rollout buffer,          │
│ training loop.                                       │
├─────────────────────────────────────────────────────┤
│ LAYER 2 — TASK / MDP DEFINITION (mjlab)              │
│ observations, actions, rewards, terminations,        │
│ curriculum. THIS is where almost all your            │
│ engineering effort goes for a specific robot.        │
├─────────────────────────────────────────────────────┤
│ LAYER 1 — PHYSICS (MuJoCo / MuJoCo-Warp)             │
│ robot body (MJCF), contact, actuators.               │
│ Doesn't know what "reward" even means.               │
└─────────────────────────────────────────────────────┘
```

- **Layer 1 is done for you.** MuJoCo is a physics engine — it takes gravity, friction, and the torques your policy commands, and computes where every joint ends up next. You never write this math yourself.
- **Layer 2 is where you'll actually spend your engineering time.** mjlab's only job is to receive Layer 1's raw physics data and pack it into tensors — observations, rewards, and done-flags — so Layer 3 can treat everything as one giant math problem. This packaging is *done for you* by mjlab; your job is deciding *what* goes into those tensors (which observations, which reward terms, which termination conditions) for your specific robot.
- **Layer 3 you will not touch**, at least not by editing its internals. You already built this mentally in the HuggingFace course — PPO loss, actor-critic, all of it. Here it's the same math, just running on batched tensors from Layer 2 instead of one CartPole at a time.

Once Layer 1's dynamics are packed into tensors by Layer 2, you can treat the robot's physics as a black box. From that point on, it's just an MDP: observations in, actions out, reward as feedback — tensor in, tensor out.

---

## 3. The MDP, mapped onto a quadruped

You already know the reinforcement learning loop: observation → policy → action → reward. This section maps each piece onto actual quadruped numbers, and fills in two details that don't come up in toy environments but matter a lot here.

### Observation and action tensors

A typical observation for quadruped locomotion is built from things like:

```
12 joint positions + 12 joint velocities + base orientation + previous action + ...
                                                        ≈ 45 numbers total
```

That collection of numbers is your observation, `o_t`. The policy is a function that maps it to an action:

```
a_t = π(o_t)          "actor"
v_t = V(o_t)          "critic" — estimates how good o_t is
```

Both `π` (the actor) and `V` (the critic) are neural networks — this is the "actor-critic" pair you already know from the HuggingFace course. At every timestep, you get exactly three tensors out of this loop: `o_t`, `a_t`, `r_t`.

### Two details that don't show up in toy environments

**1. The policy outputs a *distribution*, not a single number.** During training, the actor network outputs the mean of a Gaussian distribution over actions (with a learned or fixed spread). The actual action `a_t` is *sampled* from that distribution — this sampling is what gives PPO the exploration it needs. At deployment (or during evaluation/`play`), you skip the sampling and just use the mean directly, so the robot's behavior is deterministic.

**2. The action is not a torque.** `a_t` is usually a small offset added to a default standing joint position, scaled by a fixed constant (the "action scale"). That target position is then handed to a low-level PD controller — built into the simulator, not something you write — which converts "go to this joint position" into the actual torque needed to get there. So the pipeline for one joint looks like:

```
policy output  →  scaled offset  →  target joint position  →  PD controller  →  torque  →  physics
```

This matters because it explains something that otherwise looks confusing: reward functions (Section 8) penalize *torques* and *joint velocities* directly, even though the policy never outputs either of those — they're downstream consequences of the position targets it does choose.

### Why you need both an actor and a critic

The actor is the only thing that ever gets deployed — it's the policy that runs on the real robot. The critic exists purely to make training more efficient: it estimates the value of a state so PPO can compute *advantages* (how much better an action was than expected), which is what actually drives the policy update. Once training is done, the critic is discarded. This split — one network you keep, one you throw away — is worth remembering because it also means the critic is allowed to "cheat": some setups feed it privileged information the actor never sees (like true friction values), since it only needs to exist during simulated training.

---

## 4. Scaling up: parallel environments and tensors

Training on one robot isn't enough — one robot gives you one trajectory's worth of data per episode, which is far too little and far too correlated to train a stable policy from. The fix is to run many independent copies of the same environment simultaneously.

- Each independent copy — one simulated robot with its own physics state — is called an **environment**.
- It's common to train with **as many as 2048 or 4096 environments in parallel.**
- This is computationally heavy, but it's exactly what GPUs are built for — every environment steps its physics and computes its own observation/action/reward at the same time, as one batched array operation, not thousands of separate loops.

Each environment produces its own observation, action, and reward at every timestep, and all of them get packed together into tensors. If your observation has 45 numbers and you're running 4096 environments, your observation tensor has shape `(4096, 45)` — this is what "batched" means in this context. Same logic applies to the action tensor `(4096, num_actions)` and the reward tensor `(4096,)`.

One more detail worth knowing: when an individual environment's episode ends (the robot falls, or a time limit is hit), it doesn't stop the whole batch — it just resets *that one* environment in place and keeps going. The done-flags for a batch are a boolean tensor of shape `(4096,)`, and RSL_RL handles the per-environment auto-reset for you.

---

## 5. PyTorch's role, briefly

PyTorch is built specifically to make this "thousands of parallel copies" workflow efficient — that's the entire point of the library: fast mathematical operations on large batched tensors, on GPU.

The other thing it does is remove the need to hand-write backpropagation. You decide the architecture of your neural network (how many layers, what size, MLP vs. something else) and the loss function (PPO's clipped surrogate objective, already implemented in RSL_RL). PyTorch computes the gradients and updates the weights for you — you never write derivative code by hand.

So, in short: thousands of environments in parallel gets you enough data to converge quickly, and PyTorch is what makes running the math on all of that data — and training the networks on it — computationally practical.

---

## 6. Quick reference

| Term you know | What it becomes here |
|---|---|
| One environment, one episode at a time | Thousands of copies running simultaneously, batched into one GPU tensor |
| `obs` = numpy array, shape `(obs_dim,)` | `obs` = tensor, shape `(num_envs, obs_dim)`, on GPU |
| `action` = scalar or small array | `action` = tensor `(num_envs, num_actions)` — continuous joint targets, not discrete choices |
| `reward` = one float from the env | `reward` = tensor `(num_envs,)`, a weighted sum of several hand-designed terms |
| `done` = bool, you call `reset()` | `dones` = bool tensor `(num_envs,)`; finished environments auto-reset in place |
| Policy network (MLP, obs→action) | `ActorCritic` module — actor MLP + critic MLP, same idea at larger scale |
| A collected trajectory | `RolloutStorage` — a pre-allocated GPU buffer of shape `(steps_per_env, num_envs, ...)` |
| PPO clipped-surrogate update | The same math, batched harder — not different |
| "Train an episode, then repeat" | `OnPolicyRunner`: collect steps across all parallel envs → one PPO update → log → repeat |

---

## 7. Making training robust: curriculum, domain randomization, and events

These three terms get used loosely and interchangeably, but they're doing three distinct jobs. Keeping them separate will save you confusion later when you're reading a task config.

| Concept | What it does | Why |
|---|---|---|
| **Curriculum** | Progressively increases task difficulty as the policy improves — e.g. harder terrain, wider commanded-velocity ranges | Lets the policy learn easy behavior first, then build on it, instead of facing the hardest case from step one |
| **Domain randomization** | Randomizes physical parameters at each reset — friction, mass, motor strength, sensor noise | Prevents the policy from overfitting to one exact simulated physics, which is what makes it more likely to transfer to a real robot later |
| **Events** | Mid-episode disturbances applied while the robot is already walking — e.g. randomly pushing it in a random direction at random intervals | Forces the policy to learn active balance recovery, not just a single memorized gait |

All three exist for the same underlying reason: a policy that only ever sees one clean, unperturbed version of the world will fail the moment reality deviates from it even slightly.

---

## 8. Reward engineering

### The idea

A reward is just feedback to your optimizer (the policy), telling it which decisions to repeat and which to avoid. That's the whole concept — the complexity is entirely in *designing* good feedback, not in the math of using it.

### Breaking down "walk forward" into reward terms

Take a goal like "make the quadruped walk forward" and break it into the individual conditions that actually define success:

1. It should walk with an upright posture.
2. It should not trip over its own feet.
3. It should keep its torso at a target height — not on the floor, not too high.
4. It should track each of its 12 joints' commanded angle and velocity.
5. It should walk forward at the commanded speed — no more, no less.

...and many more factors beyond this. Each of these becomes its own **reward term**, and the terms are combined into a single scalar reward at every timestep as a weighted sum:

```
r_t = Σ  w_i · reward_term_i(o_t, a_t)
```

That combined `r_t` is what the policy actually optimizes against — it's the same reward variable from the PPO loss you already know, just constructed from several hand-designed pieces instead of coming for free from the environment (like a game's score would).

### What this looks like in real code

These are realistic examples of the kind of per-term reward functions you'll find in RSL_RL-style training codebases:

```python
def _reward_lin_vel_z(self):
    # Penalize z-axis base linear velocity (discourage bouncing)
    return torch.square(self.simulator.base_lin_vel[:, 2])

def _reward_ang_vel_xy(self):
    # Penalize xy-axis base angular velocity (discourage wobbling)
    return torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)

def _reward_orientation(self):
    # Penalize a non-flat base orientation (keeps the robot upright)
    return torch.sum(torch.square(self.simulator.projected_gravity[:, :2]), dim=1)

def _reward_base_height(self):
    # Penalize deviation from the target torso height
    base_height = torch.mean(
        self.simulator.base_pos[:, 2].unsqueeze(1) - self.simulator.measured_heights, dim=1
    )
    return torch.square(base_height - self.cfg.rewards.base_height_target)

def _reward_torques(self):
    # Penalize torque usage (discourages wasted energy / jerky actuation)
    return torch.sum(torch.square(self.simulator.torques), dim=1)

def _reward_dof_vel(self):
    # Penalize high joint velocities (encourages smoother motion)
    return torch.sum(torch.square(self.simulator.dof_vel), dim=1)

def _reward_dof_power(self):
    # Penalize power consumption (torque × velocity)
    return torch.sum(torch.abs(self.simulator.torques * self.simulator.dof_vel), dim=1)
```

A pattern worth noticing: almost every term is a `torch.square(...)` or `torch.abs(...)` of some quantity you want to keep *small* — squaring makes the penalty grow faster the further the robot strays, and guarantees the penalty is never negative. Each function here is a guardrail against one specific failure mode; remove the height penalty, for instance, and you'll often see the policy find a "walking" gait that crouches or crawls instead, because nothing told it not to.

### How the reward is actually used

These per-term rewards are summed with their weights into `r_t`, which feeds directly into the same PPO loss computation you already know from the HuggingFace course — it's used to compute returns and advantages, which drive the policy gradient update. Nothing about the *algorithm* changes; only the source of `r_t` gets more elaborate.

---

## 9. How to actually use RSL_RL — extend, don't rewrite

RSL_RL already has full implementations of computing rewards, computing observations, applying actions to the environment (the `step()` function) — all of it. Your job is not to write any of this from scratch; it's to *customize* the small parts specific to your robot: a new reward term, an additional observation, a different termination condition.

The important rule: **don't directly edit the pre-written code.** Instead, subclass and override — import or inherit the provided classes, then override just the specific methods or variables you need to change. This is the standard, safest, and most maintainable way to work with a framework like this, and it also means your changes survive if the underlying library gets updated.

One clarification worth making explicit, tying back to the three-layer picture in Section 2: this "extend by inheriting" pattern applies at **Layer 2** — the task/environment definition (in mjlab, or an equivalent env wrapper). RSL_RL's own PPO/actor-critic code in Layer 3 is something you *configure* (learning rate, network sizes, rollout length) rather than subclass or edit — you're customizing the problem definition, not the algorithm that solves it.

---

## 10. A worked example to study

The clearest way to see this pattern in real code is to look at a complete, working implementation. Here's one built from scratch on top of RSL_RL, using Genesis as the simulator instead of MuJoCo — the concepts transfer directly:

**Repository:** [github.com/aceofspades07/pip-loco](https://github.com/aceofspades07/pip-loco)

Specifically, read `envs/genesis_wrapper.py`. That file shows exactly how a custom MDP environment gets built on top of RSL_RL's framework — the same inherit-and-override pattern described in Section 9, applied concretely. The rest of the repo shows the neural network definitions and the full PPO training setup end to end.

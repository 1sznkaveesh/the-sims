# Pupper locomotion — implementation roadmap

*A continuation of "RSL_RL quadruped teaching session." That session covered the concepts (see the concepts primer). This document is the actual track: the concrete, checkable implementation tasks that take you from a fresh machine to a Stanford Pupper walking in MuJoCo simulation via mjlab + RSL_RL.*

**Scope, explicitly:** this roadmap ends at "Pupper walks forward on flat ground in simulation." Sim-to-real, rough terrain, and anything past that is out of scope for now — Phase 9 just names what comes next so the team isn't surprised by it later.

**How to use this document:** work top to bottom. Each phase has a goal, a checklist of tasks, links to the actual documentation (this doc intentionally doesn't re-explain what the docs already explain), and a "definition of done" — don't move to the next phase until you can honestly check that box. Each phase is also tagged with who it needs.

---

## Phase 0 — Environment check

**Goal:** confirm you can actually run this before writing any project-specific code.

- [ ] Confirm GPU access. mjlab **requires an NVIDIA GPU for training**; macOS is supported for evaluation only. If you don't have one locally, use the hosted Google Colab linked in the mjlab README — no local setup required.
- [ ] Create a free [Weights & Biases](https://wandb.ai) account. mjlab's `play` command fetches trained checkpoints via `--wandb-run-path`, so you'll want this before Phase 1.
- [ ] Confirm `uv` is installed (mjlab's install path is uv-based — see the Installation Guide linked below).

**Who this needs:** self-serve.

---

## Phase 1 — Install mjlab and run the shipped example

**Goal:** prove the whole stack works end to end on a known-good robot before touching anything Pupper-specific.

- [ ] Install mjlab following the README / Installation Guide:
  [github.com/mujocolab/mjlab](https://github.com/mujocolab/mjlab) · [mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab)
- [ ] Train the shipped Go1 quadruped task (scale `num-envs` down if you're not on a large GPU):
  ```
  uv run train Mjlab-Velocity-Flat-Unitree-Go1 --env.scene.num-envs 512
  ```
- [ ] Watch the reward curves climb in TensorBoard or W&B.
- [ ] Evaluate a policy — first with garbage actions, then with your trained one, so you can see the difference:
  ```
  uv run play Mjlab-Velocity-Flat-Unitree-Go1 --agent random
  uv run play Mjlab-Velocity-Flat-Unitree-Go1 --wandb-run-path <your-run>
  ```

**Definition of done:** you can train and play the shipped Go1 task with no errors, and for any piece of output on screen you can say which layer (1/2/3, from the primer) it belongs to.

**Who this needs:** self-serve. If installation itself breaks (driver/CUDA issues), that's worth a quick group message rather than everyone debugging solo.

---

## Phase 2 — Read the Go1 task config end to end

**Goal:** know exactly which files you'll be writing the Pupper equivalent of later. No training in this phase — just reading.

- [ ] Locate the Go1 velocity task's config and find each manager: scene, observations, actions, rewards, terminations, curriculum/events.
- [ ] For each manager, write one sentence in your own words describing its job. This becomes your personal reference for Phase 6.
- [ ] Optional: use mjlab's `export-scene` CLI to dump the Go1 scene/asset to a directory for closer inspection.

**Definition of done:** without looking anything up, you can explain what each manager in the Go1 config does.

**Who this needs:** self-serve.

---

## Phase 3 — Build intuition by breaking Go1 on purpose

**Goal:** see reward/observation changes cause visible gait changes, on a robot that's known to work — before you have your own possibly-broken robot to debug against.

- [ ] Change one reward term's weight (e.g., disable the base-height penalty), retrain briefly, and describe what changes about the gait.
- [ ] Remove one observation (e.g., previous action) and note what breaks or degrades.
- [ ] Share what you found with the rest of the group — everyone doing this on the same robot means you can compare notes directly.

**Definition of done:** the group can predict, roughly, what a given reward/observation change will do before running it.

**Who this needs:** self-serve.

---

## Phase 4 — Get Pupper's robot description into MuJoCo format

**Goal:** one physically sane Pupper MJCF file, confirmed working, entirely independent of RL. This is a robotics/asset-prep task, not an RL task — don't debug a bad robot model from inside a training run.

- [ ] Confirm which Pupper hardware revision you're targeting — this decides your starting repo:
  - **Pupper v3:** [github.com/G-Levine/pupper_v3_description](https://github.com/G-Levine/pupper_v3_description) — includes a scripted URDF → MuJoCo XML pipeline (`create_mujoco_xml.py`, with `--mjx` and `--fixed` options).
  - **Original/older Pupper:** [github.com/stanfordroboticsclub/StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped) for the reference design, or [github.com/chandyalex/stanford_pupper_description](https://github.com/chandyalex/stanford_pupper_description) for an existing URDF to convert.
- [ ] Follow that repo's conversion instructions to produce an MJCF.
- [ ] Load the resulting MJCF in the **native MuJoCo viewer directly** (outside mjlab entirely) and confirm: the robot doesn't clip through the floor or explode, joint limits look sane, it settles under gravity without flying apart.

**Definition of done:** Pupper stands (or at least settles calmly) in the MuJoCo viewer under gravity, with no RL involved yet.

**Who this needs:** self-serve, but flag it to the instructor if the model behaves obviously wrong (limbs pass through the body, joints spin freely) — that's usually a units or joint-limit issue in the source URDF, not something to debug blind.

---

## Phase 5 — Wire Pupper into mjlab as a new task

**Goal:** a task ID (e.g. `Mjlab-Velocity-Flat-Pupper`) that runs — falling over immediately is a perfectly fine outcome at this stage. This phase is the direct parallel of what you saw done for Go1 in the session.

- [ ] Study [github.com/mujocolab/anymal_c_velocity](https://github.com/mujocolab/anymal_c_velocity) — a complete, real example of bolting a new quadruped onto mjlab's velocity-tracking task. This is your blueprint; follow its structure rather than starting from a blank file.
- [ ] Register Pupper's MJCF as the scene's entity, and start from the Go1/ANYmal observation and action term sets as your baseline (don't design a new MDP from scratch — copy, then adapt).
- [ ] Adjust the action space to Pupper's 12 joints — check joint ordering, limits, and control mode (position vs. torque) against your MJCF from Phase 4.
- [ ] Sanity-check with the built-in agents before any training:
  ```
  uv run play Mjlab-Velocity-Flat-Pupper --agent zero
  uv run play Mjlab-Velocity-Flat-Pupper --agent random
  ```

**Definition of done:** the zero-action and random-action agents both run for a full episode, across all parallel environments, with no crashes.

**Who this needs:** **instructor involved.** This is the one step with real plumbing risk (asset registration, action/observation wiring) — budget time to pair on this rather than solving it solo.

---

## Phase 6 — Retune the MDP for Pupper's scale

**Goal:** rewards, terminations, and curriculum tuned for Pupper's actual mass and size — not Go1's defaults left unchanged.

- [ ] Copy the Go1 reward term list as your starting point (see Section 8 of the concepts primer).
- [ ] Retune weights — Pupper is far lighter and lower-torque than Go1, so energy/torque penalty weights especially need rescaling, or they'll dominate the loss.
- [ ] Set a target base height appropriate to Pupper's leg length.
- [ ] Check termination conditions (fall-detection height/orientation thresholds) actually match Pupper's geometry, not Go1's.

**Definition of done:** a written list (in your repo's README or config comments) of every reward term and why its weight was chosen for Pupper specifically.

**Who this needs:** self-serve, using the concepts primer's reward-engineering section as your reference.

---

## Phase 7 — First training run: flat ground, velocity tracking

**Goal:** a policy that walks forward without falling, on flat ground, for a sustained rollout.

- [ ] Start with a narrow commanded-velocity range (forward-only, low speed) before widening anything.
- [ ] Train while watching per-term reward curves, not just total reward — a flat total can hide one term dominating the others.
- [ ] `play` the checkpoint periodically during training rather than waiting until it finishes.

**Definition of done:** a policy that reliably walks forward on flat ground, across most parallel environments, for your full episode length.

**Who this needs:** self-serve.

---

## Phase 8 — Evaluate, debug, iterate

**Goal:** catch and fix gait pathologies before calling it done.

- [ ] Watch for common failure patterns: foot dragging, excessive energy use, instability at higher commanded speeds, limping to one side.
- [ ] For each pathology you see, trace it back to a missing or under/over-weighted reward term (or a bad termination condition), adjust, and retrain.
- [ ] Only after flat-ground walking is genuinely solid, consider widening the curriculum (speed range, then terrain).

**Definition of done:** the team agrees the gait looks physically reasonable, with a saved checkpoint you'd be comfortable showing outside the club.

**Who this needs:** self-serve, group review recommended before declaring a checkpoint "final."

---

## Phase 9 — What comes after this roadmap (not in scope yet)

Named here so the team knows the shape of what's next, without turning this doc into a task list for it:

1. **Robustness:** domain randomization and events (Section 7 of the primer), then rough-terrain curriculum.
2. **Sim-to-sim validation:** does the policy still work if you change simulator settings/parameters slightly?
3. **Sim-to-real:** deployment onto the physical Pupper.

Don't start on these until Phase 8 is genuinely done — a policy that isn't solid on flat ground won't get more solid by adding terrain.

---

## Reference index

| Resource | Link |
|---|---|
| mjlab (repo + README) | https://github.com/mujocolab/mjlab |
| mjlab documentation | https://mujocolab.github.io/mjlab |
| Custom-robot integration blueprint (ANYmal) | https://github.com/mujocolab/anymal_c_velocity |
| Additional task examples (mjlab_playground) | https://github.com/mujocolab/mjlab_playground |
| Pupper v3 description + MJCF conversion | https://github.com/G-Levine/pupper_v3_description |
| Original Stanford Pupper design | https://github.com/stanfordroboticsclub/StanfordQuadruped |
| Stanford Pupper URDF (alternate source) | https://github.com/chandyalex/stanford_pupper_description |
| Weights & Biases | https://wandb.ai |

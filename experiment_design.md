# Autonomous Learning Without a Pre-Given Knowledge Base
### Experiment design document

---

## 1. The core hypothesis

Human "learning" about the physical world is largely not instruction — it's the discovery of constraints through interaction, followed by an internal model that predicts them. A child doesn't learn gravity from a fact; it learns gravity from prediction error (things fall when let go) and then stops being surprised by it.

**Hypothesis:** an AI agent placed in a lawful, constraint-enforcing environment, with no task reward and no fed-in facts, and optimized purely to reduce its own prediction error about the consequences of its actions, will spontaneously discover the environment's constraints (physics, resource scarcity, day/night cycles, object permanence) the same way — without those constraints ever being described to it.

This reframes "no sensors available" as a non-blocker: the requirement isn't a physical body, it's an environment that enforces its own rules independent of the agent. A simulator with real collision, gravity, and resource logic satisfies that; text describing those rules does not (that's testimony, not discovery).

---

## 2. Why a simulated environment substitutes for sensors

Three properties matter, and simulators can supply all three without any physical hardware:

1. **Lawfulness** — the environment behaves consistently according to fixed rules the agent never sees directly (a physics/game engine, not a human).
2. **Consequence** — the agent's actions change its future observations in ways it must predict to do well (self-supervision signal).
3. **Novel, unlabeled state space** — the agent isn't handed categories or facts; it only gets raw sensory frames.

Given that, a procedurally generated game world is a legitimate stand-in for "the world it is encountering." It's not full embodiment (single modality, no smell/touch/proprioceptive richness), but it isolates the mechanism you actually want to test — self-supervised constraint discovery — from confounds like sensor noise and robot hardware cost.

**Environment used here: [Crafter](https://github.com/danijar/crafter).** A top-down 2D survival environment (procedurally generated worlds, day/night cycle, hunger/thirst, resource gathering, crafting, threats). Chosen because:
- It has genuine, enforced constraints analogous to your examples (can't walk through trees/stone; limited daylight; food and water are finite and depleting).
- It ships built-in "achievements" (collect wood, drink water, wake up, etc.) that are *never exposed to the agent* — they exist purely so an outside observer can check whether the agent's undirected behavior stumbled into meaningful, game-consistent milestones. This gives you a clean, pre-built evaluation probe without designing one from scratch.
- It's small enough to run on a laptop CPU (64×64 pixel observations), so iteration is fast.

Other environments worth knowing about for later stages: **Craftax** (a faster, GPU-native reimplementation of Crafter), **MuJoCo/PyBullet** (continuous-control physics, better for testing "can't fly," "can't pass through walls" as literal rigid-body constraints), **NetHack/MiniHack** (much richer rule systems, harder).

---

## 3. Why the learning signal can't be reward or labels

Task reward and labeled data are both external knowledge bases in disguise — someone still had to define "good" or supply the category. The agent needs a self-supervised signal that comes from the environment's own dynamics.

**Mechanism used: Intrinsic Curiosity Module (ICM)**, after Pathak et al., 2017 ("Curiosity-driven Exploration by Self-supervised Prediction"). Three parts:

- **Encoder** — compresses a raw observation into a feature vector. This is the agent's "sense."
- **Inverse model** — given features of two consecutive observations, predict the action that connected them. This trains the encoder to represent only the parts of the world the agent's own actions affect (filters out irrelevant background so curiosity isn't captured by things the agent can't control — the "noisy TV" problem).
- **Forward model** — given the current feature vector and an action, predict the next feature vector. **The prediction error of this model is the agent's reward.** Surprise is the only thing being optimized for.

No environment reward, no achievement signal, no text description of rules is ever given to the policy or either model. Everything the agent "knows" has to come from having been surprised and having updated to reduce that surprise.

---

## 4. Independent variables to test (this is where your open questions become experiments)

| Variable | Conditions to compare | What it tests |
|---|---|---|
| **Modality richness** | Vision only vs. vision + agent-internal state (e.g. an inventory/proprioception channel) | Your question about single-sense learning: does restricting to one sense measurably slow or cap constraint discovery, holding the environment fixed? |
| **Curriculum** | Fixed full-complexity world from step 0 vs. staged complexity (e.g., start with a static world, then introduce day/night, then introduce threats) | Whether developmental-style staging (mirrors infant learning literature) speeds convergence or changes what gets learned first |
| **Intrinsic objective** | Forward-model prediction error (ICM) vs. novelty/state-visitation counts vs. empowerment (control over future states) | Which self-supervised signal produces the most "human-like" discovery order |
| **Environment stochasticity** | Deterministic seed vs. procedurally varied worlds each episode | Whether the agent learns a generalizable model of constraints vs. memorizing one map |

Recommended first cut: hold curriculum and objective fixed, vary **modality richness** only — it's the question you raised explicitly, it's a one-line code change (see `Encoder` in `icm_agent.py`), and it directly produces a comparable metric (forward-model prediction error curve, achievement-probe count) across conditions.

---

## 5. Evaluation methodology ("how do we know it learned something")

Because there's no reward to plot, evaluation has to probe the *internal model*, not behavior alone:

1. **Prediction-error curve over training** — should trend downward as the world becomes less surprising to the agent. A flattening curve on a fixed world = the agent has captured the deterministic parts of that world's dynamics.
2. **Achievement/milestone probe (external, read-only)** — log how many of Crafter's built-in achievements get unlocked over time purely from curiosity-driven behavior, with zero achievement signal fed to the agent. This is analogous to checking whether a human infant "figures out" object permanence without being taught the term — you're checking for the behavior, not asking if it can recite the rule.
3. **Held-out consequence prediction** — after training, freeze the world-model and test it on action/observation pairs it wasn't specifically optimized on; measure forward-model error. This checks whether it built a generalizable model of the environment's laws vs. memorized specific transitions.
4. **Constraint-violation test** — deliberately query the forward model with an action that should be physically impossible in the environment (e.g., "walk into stone") and check whether its predicted next-state resembles "no movement occurred" — i.e., has it implicitly learned the constraint exists.

---

## 6. Phased roadmap

**Phase 0 (done in this session):** working ICM + Crafter prototype, single modality (vision), no curriculum, verified to run and produce sensible loss curves plus achievement unlocks over a short run.

**Phase 1:** run to convergence (tens of thousands of steps) on your own machine; plot prediction-error curve and achievement-probe timeline; establish baseline.

**Phase 2:** modality-richness ablation (Section 4) — compare vision-only vs. vision+state, same seed, same step budget.

**Phase 3:** curriculum ablation — staged-complexity world vs. full-complexity from the start.

**Phase 4 (optional, higher effort):** move from Crafter to a physics engine (PyBullet/MuJoCo) to test literal rigid-body constraints ("can't fly," "can't pass through walls") as continuous control rather than a grid game, closer to your original examples.

---

## 7. Anchoring literature (for context, not required reading to run the prototype)

- **Bengio et al.** — work on causal/agentic world models, GFlowNets, and the broader "System 2" research agenda motivating this kind of self-supervised, out-of-distribution generalization.
- **LeCun** — Joint Embedding Predictive Architectures (JEPA), the broader case for self-supervised world-models as the path past supervised/labeled learning.
- **Pathak et al., 2017** — the ICM algorithm used here.
- **Oudeyer** — developmental robotics, curiosity-driven infant learning; the basis for the curriculum variable in Section 4.
- **Friston** — active inference / free-energy principle; an alternative theoretical framing of "surprise minimization" worth knowing if you want to compare intrinsic-objective formulations later.
- **Hafner (Crafter benchmark paper)** — the environment's original design rationale and baseline results.

---

## 8. Honest limitations

- Single modality (vision) is a real simplification; Section 4's ablation is designed to measure, not hand-wave past, what that costs.
- Crafter is still a human-designed environment — the "laws" are game rules, not physics. This is a reasonable proxy, not a claim of full generality; Phase 4 moves toward literal physical constraints for that reason.
- ICM's forward-model-error reward is known to be gameable by genuinely stochastic elements the agent can't control (the "noisy TV" problem); the inverse model mitigates but doesn't eliminate this, which is why the held-out consequence prediction test (Section 5.3) matters — it distinguishes real model-building from reward-hacking the curiosity signal itself.

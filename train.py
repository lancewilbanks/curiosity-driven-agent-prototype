"""
Curiosity-only training loop in Crafter.

IMPORTANT re: the experiment's premise (no external knowledge base):
  - The agent's `act()` and both losses (ICM, actor-critic) never see
    Crafter's reward signal or its `info['achievements']` dict.
  - `info['achievements']` is read ONLY here, after the agent has already
    acted, purely to log an external probe: "did the agent's undirected,
    curiosity-only behavior stumble into game-defined milestones (collect
    wood, drink water, wake up, etc.) on its own, the way an infant
    discovers what its environment permits without being told the rules?"
    This is analysis, not training signal.

Run: python3 train.py --steps 4000
"""
import argparse
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F

import crafter
from icm_agent import ICMAgent


def preprocess(obs_hwc_uint8):
    # (H,W,C) uint8 -> (C,H,W) float32 in [0,1]
    x = torch.from_numpy(obs_hwc_uint8).float() / 255.0
    return x.permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rollout_len", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default="run_log.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = crafter.Env(seed=args.seed)
    n_actions = env.action_space.n
    device = "cpu"

    agent = ICMAgent(in_channels=3, n_actions=n_actions, lr=args.lr, device=device)

    obs = env.reset()
    obs_t = preprocess(obs)

    cumulative_achievements = {}
    total_steps = 0
    t0 = time.time()

    with open(args.log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "steps", "mean_intrinsic_reward",
                          "icm_loss", "policy_loss", "value_loss",
                          "entropy", "new_achievements_unlocked", "wall_s"])

        update = 0
        while total_steps < args.steps:
            obs_buf, act_buf, logp_buf, val_buf, ent_buf = [], [], [], [], []

            for _ in range(args.rollout_len):
                obs_batch = obs_t.unsqueeze(0)
                action, logp, value, entropy = agent.act(obs_batch)

                next_obs, _reward, done, info = env.step(action.item())

                # --- external probe only, not fed back into the agent ---
                for k, v in info.get("achievements", {}).items():
                    if v > cumulative_achievements.get(k, 0):
                        cumulative_achievements[k] = v
                # ----------------------------------------------------------

                obs_buf.append(obs_t)
                act_buf.append(action.squeeze(0))
                logp_buf.append(logp.squeeze(0))
                val_buf.append(value.squeeze(0))
                ent_buf.append(entropy.squeeze(0))

                obs_t = preprocess(next_obs if not done else env.reset())
                total_steps += 1
                if total_steps >= args.steps:
                    break

            obs_stack = torch.stack(obs_buf)                      # (T,C,H,W)
            next_obs_stack = torch.cat([obs_stack[1:], obs_t.unsqueeze(0)], dim=0)
            act_stack = torch.stack(act_buf)                      # (T,)
            logp_stack = torch.stack(logp_buf)
            val_stack = torch.stack(val_buf)
            ent_stack = torch.stack(ent_buf)

            intrinsic_r, icm_loss = agent.intrinsic_reward_and_icm_loss(
                obs_stack, act_stack, next_obs_stack)

            # n-step bootstrap return using the agent's own value estimate
            with torch.no_grad():
                _, bootstrap_v = agent.ac(obs_t.unsqueeze(0))
            returns = torch.zeros_like(intrinsic_r)
            R = bootstrap_v.squeeze(0)
            for i in reversed(range(len(intrinsic_r))):
                R = intrinsic_r[i] + args.gamma * R
                returns[i] = R
            advantage = returns - val_stack.detach()

            policy_loss = -(logp_stack * advantage).mean()
            value_loss = F.mse_loss(val_stack, returns)
            entropy_bonus = ent_stack.mean()
            ac_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            agent.ac_opt.zero_grad()
            ac_loss.backward()
            agent.ac_opt.step()

            agent.icm_opt.zero_grad()
            icm_loss.backward()
            agent.icm_opt.step()

            update += 1
            writer.writerow([update, total_steps,
                              float(intrinsic_r.mean()),
                              float(icm_loss.item()),
                              float(policy_loss.item()),
                              float(value_loss.item()),
                              float(ent_stack.mean().item()),
                              len(cumulative_achievements),
                              round(time.time() - t0, 1)])
            f.flush()

            if update % 10 == 0 or total_steps >= args.steps:
                print(f"update {update:4d} | steps {total_steps:5d} | "
                      f"mean intrinsic r {intrinsic_r.mean().item():.4f} | "
                      f"icm_loss {icm_loss.item():.4f} | "
                      f"achievements probed so far: {sorted(cumulative_achievements.keys())}")

    print("\nDone.")
    print(f"Total wall time: {time.time()-t0:.1f}s")
    print(f"Achievements the curiosity-only agent stumbled into (probe, not reward):")
    for k, v in sorted(cumulative_achievements.items()):
        print(f"  {k}: unlocked (count={v})")


if __name__ == "__main__":
    main()

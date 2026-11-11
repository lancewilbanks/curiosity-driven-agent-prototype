"""
Intrinsic Curiosity Module (ICM) agent, after Pathak et al. 2017
("Curiosity-driven Exploration by Self-supervised Prediction").

Design intent (see docs/experiment_design.md for the full rationale):
  - The agent receives NO extrinsic task reward and NO hand-fed facts about
    the world. Its only input is raw pixels from the environment.
  - It is rewarded purely for how *wrong* its own forward model was about
    the consequences of its actions (prediction error) -- i.e. it is
    rewarded for encountering things it doesn't yet understand, exactly
    the "falling teaches gravity" logic from the design doc.
  - An inverse model (predict the action taken from consecutive feature
    embeddings) is trained jointly so the encoder learns to represent only
    the parts of the observation the agent's own actions can influence,
    which keeps curiosity from being captured by background noise the
    agent can't control (the "noisy TV" problem).

Nothing in this file references crafter-specific semantics (wood, stone,
achievements, etc.) -- the agent never sees or is told about game
mechanics. Those are only read out afterward, externally, as an evaluation
probe (see train.py), never fed back into the agent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Maps a raw (C,H,W) sensory frame to a compact feature vector.

    This is the agent's only sense in this prototype (vision). Swapping,
    restricting, or adding encoders here is how the single-sense vs.
    multi-sense experiment variable (design doc, Track B) gets implemented.
    """

    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            n_flat = self.net(dummy).view(1, -1).shape[1]
        self.fc = nn.Linear(n_flat, feat_dim)

    def forward(self, x):
        z = self.net(x)
        z = z.view(z.size(0), -1)
        return self.fc(z)


class InverseModel(nn.Module):
    """Predicts a_t from (phi(s_t), phi(s_t+1)). Trains the encoder to keep
    only action-relevant features -- the agent's model of 'what my actions
    do', not a generic image compressor."""

    def __init__(self, feat_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, phi_s, phi_s2):
        return self.net(torch.cat([phi_s, phi_s2], dim=-1))


class ForwardModel(nn.Module):
    """Predicts phi(s_t+1) from (phi(s_t), a_t). The prediction error of
    this module IS the agent's intrinsic reward -- literally 'how
    surprised was I by the consequence of my action'."""

    def __init__(self, feat_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_actions, 256), nn.ReLU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, phi_s, action):
        a_onehot = F.one_hot(action, self.n_actions).float()
        return self.net(torch.cat([phi_s, a_onehot], dim=-1))


class ActorCritic(nn.Module):
    """Policy + value head, trained ONLY on intrinsic (curiosity) reward.
    Shares no gradient with the ICM encoder (uses its own encoder) so the
    policy can't shortcut curiosity by degenerating the world-model."""

    def __init__(self, in_channels, n_actions, feat_dim=256):
        super().__init__()
        self.encoder = Encoder(in_channels, feat_dim)
        self.pi = nn.Linear(feat_dim, n_actions)
        self.v = nn.Linear(feat_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.pi(z), self.v(z).squeeze(-1)


class ICMAgent:
    def __init__(self, in_channels, n_actions, feat_dim=256, lr=1e-4,
                 beta=0.2, eta=0.5, device="cpu"):
        """
        beta: weight between inverse-model loss and forward-model loss
              when training the ICM (Pathak et al. use 0.2 by default).
        eta:  scaling applied to forward-model error to form the intrinsic
              reward (keeps reward magnitude in a sane range for the RL
              update).
        """
        self.device = device
        self.encoder = Encoder(in_channels, feat_dim).to(device)
        self.inverse = InverseModel(feat_dim, n_actions).to(device)
        self.forward_model = ForwardModel(feat_dim, n_actions).to(device)
        self.ac = ActorCritic(in_channels, n_actions, feat_dim).to(device)
        self.beta = beta
        self.eta = eta

        icm_params = (list(self.encoder.parameters()) +
                      list(self.inverse.parameters()) +
                      list(self.forward_model.parameters()))
        self.icm_opt = torch.optim.Adam(icm_params, lr=lr)
        self.ac_opt = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def act(self, obs_t):
        """obs_t: (1, C, H, W) float tensor in [0,1]. Returns action, log_prob, value."""
        logits, value = self.ac(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def intrinsic_reward_and_icm_loss(self, s, a, s2):
        """s, s2: (B,C,H,W) float tensors in [0,1]; a: (B,) long tensor.
        Returns (intrinsic_reward per-sample [no grad], icm_loss [scalar])."""
        phi_s = self.encoder(s)
        phi_s2 = self.encoder(s2)

        pred_logits = self.inverse(phi_s, phi_s2)
        inverse_loss = F.cross_entropy(pred_logits, a)

        pred_phi_s2 = self.forward_model(phi_s, a)
        forward_err = F.mse_loss(pred_phi_s2, phi_s2.detach(), reduction="none").mean(dim=-1)
        forward_loss = forward_err.mean()

        icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        intrinsic_reward = (self.eta * forward_err).detach()
        return intrinsic_reward, icm_loss

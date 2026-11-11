# Curiosity-driven agent prototype (ICM + Crafter)

No task reward, no labels, no hand-fed facts about the environment. The
agent is rewarded only for reducing its own prediction error about what
happens next. See `../docs/experiment_design.md` for the full rationale.

## Setup (Windows)

1. Install Python 3.9+ from python.org — check **"Add python.exe to PATH"**
   during install.
2. Open Command Prompt / PowerShell and confirm:
   ```
   python --version
   pip --version
   ```
3. Create a project folder, put these files in it, then make a virtual
   environment:
   ```
   cd path\to\this\folder
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   If `crafter` fails to build with a `setuptools`/`distutils` error, run
   `pip install "setuptools<60" wheel` first, then retry.

(macOS/Linux: same steps, replace `venv\Scripts\activate` with
`source venv/bin/activate`.)

## Run

```
python train.py --steps 4000
```

CPU-only, no GPU required. A few thousand steps takes roughly a minute on
a typical laptop; increase `--steps` for a longer run (Phase 1 in the
design doc suggests tens of thousands of steps for a real curve). Logs
print per-update to the terminal and get written to `run_log.csv`.

Then visualize:
```
pip install matplotlib   # already in requirements.txt
python plot_results.py run_log.csv
```
This produces `results.png`: the intrinsic-reward (surprise) curve, and
how many of Crafter's built-in achievements the agent stumbled into over
time — logged only as an outside probe, never shown to the agent.

## Files

- `icm_agent.py` — the encoder, inverse model, forward model, and
  actor-critic (heavily commented with the reasoning behind each piece).
- `train.py` — the training loop; wires the agent into Crafter.
- `plot_results.py` — turns `run_log.csv` into a chart.
- `requirements.txt` — everything `pip install -r` needs.

## Where to make the changes described in the design doc's Section 4

- **Modality-richness ablation:** edit `Encoder` in `icm_agent.py` to
  accept/ignore an extra input channel (Crafter's `info` dict exposes
  inventory state you could stack in as a second "sense").
- **Curriculum ablation:** Crafter exposes world-generation knobs; simplest
  approach is running separate training phases with different seeds/settings
  and concatenating the logs.

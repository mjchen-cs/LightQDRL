# LightQDRL

Code for the paper

> **Lightweight hybrid quantum–classical decision policies for resource-constrained edge computing**
> Miaojiang Chen, Wenjing Xiao, Haiqiang Chen, Zhenrong Zhang, Guihua Luo, Athanasios Vasilakos, Ahmed Farouk

LightQDRL is a lightweight-model framework for hybrid quantum–classical reinforcement-learning policies. It combines
three modules that act on a shared hybrid backbone and therefore plug into any learning rule:

* **DQCP**, dynamic quantum-circuit pruning: removes rotations and CNOTs during training according to their saliency;
* **JCQQ**, joint classical–quantum quantization: 8-bit weights, activations, circuit angles and read-outs;
* **PEQA**, parameter-efficient quantum adaptation: rank-2 adapters, angle offsets and two scales for on-device adaptation.

The framework is applied unchanged to seven algorithms (Q-DQN, Q-DDQN, Q-Dueling DQN, Q-A2C, Q-A3C, Q-PPO, Q-TRPO)
and evaluated on a queue-aware wireless-powered mobile-edge-computing network driven by industrial video workloads
(WP-MEC), dynamic multichannel access and CartPole-v1, in simulation, on an NVIDIA Jetson Orin Nano Super edge board
and on the Origin Quantum WK_C180 superconducting processor.

**Data:** all results underlying the paper (per-seed results, checkpoints, adaptation, robustness, on-device and
hardware records) are available at https://doi.org/10.5281/zenodo.23454656.

## Repository layout

| File | Purpose |
|---|---|
| `qrl_core.py` | Hybrid backbone (`HybridNet`), exact state-vector simulator, DQCP, JCQQ, PEQA, resource accounting |
| `lqdrl_core.py` | Quantizers, circuit construction, gate counts, Qiskit transpilation statistics, utilities |
| `qrl_envs.py` | WP-MEC (`mec`; `mec_v1` = earlier version used for the hardware agent), multichannel access (`channel`), CartPole (`cartpole`), reference policies, difference rewards |
| `qrl_agents.py` | The seven algorithms on the shared backbone (`make_agent`, `ALGOS`, `ENV_HP`) |
| `optimization.py` | Bisection-based WPT / time-allocation optimizer from DROO (used by `mec_v1`) |
| `exp_qrl_bench.py` | Main benchmark: trains and evaluates every algorithm / variant / core / seed |
| `select_lightqdrl_x.py` | Pre-registered selection of LightQDRL-X on validation seeds |
| `exp_qrl_adapt.py` | Adaptation to traffic and channel shifts (frozen, full fine-tuning, PEQA) |
| `exp_qrl_robust.py` | Gate noise, finite shots, routing and closed-loop noisy evaluation |
| `measure_qrl_latency.py` | Simulated single-decision latency on the workstation |
| `measure_edge_device.py` | On-device latency, stored size, decision agreement, closed-loop score and energy (Jetson, Raspberry Pi 5, x86) |
| `origin_hw_validation.py` | Execution of the policy circuits on the Origin Quantum cloud (pyqpanda3); `--mode dryrun` runs offline |
| `quafu_hw_validation.py` | Shared hardware-validation helpers (and Quafu cloud support) |
| `origin_transpile_estimate.py` | Compiled CZ count, SWAPs and depth on the WK_C180 coupling graph (Qiskit) |
| `plot_common.py`, `plot_qrl_fig3.py` … `plot_qrl_fig7.py` | Figures 3–7; every script also writes the plotted numbers to `figures/*.csv` |
| `make_video_sizes.py` | Rebuilds `data/video_task_sizes.csv` from a folder of video clips |
| `make_data_package.py` | Builds the data deposit (zip, README, checksums) from the result folders |
| `figures_src/` | Native PowerPoint sources of Figs. 1–2 (`node make_fig1.js`, `node make_fig2.js`; needs `npm i pptxgenjs`) |
| `data/data_10.mat` | Mean channel gains of ten devices (DROO dataset, see `third_party/`) |
| `data/video_task_sizes.csv` | Sizes of the 58 industrial video clips that define WP-MEC task sizes (anonymized identifiers) |

## Installation

Python 3.10 (tested on Windows 11 and Linux; the edge measurements on aarch64 Linux).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: `pip install pyqpanda3` (Origin Quantum hardware), `pip install pyquafu` (Quafu). All training uses the CPU
with one thread per run; no GPU is required.

## Reproducing the paper

All scripts are resumable: finished runs are detected and skipped. `--workers N` runs N processes in parallel.
Results are written to `results_qrl*/`, figures to `figures/`.

```bash
# 1. Main benchmark (Figs. 3-4, Table 1)
python exp_qrl_bench.py --env mec --workers 4
python exp_qrl_bench.py --env channel --workers 4
python exp_qrl_bench.py --env cartpole --workers 4
# quantum-removal and classical controls, and their 8-bit versions (Table 2)
python exp_qrl_bench.py --env mec --cores tanh,mlp --variants base,light --workers 4
python measure_qrl_latency.py --env mec

# 2. Pre-registered selection on validation seeds 100-102 (Fig. 5a)
python exp_qrl_bench.py --env mec --seed_offset 100 --seeds 3 --workers 4
python select_lightqdrl_x.py                       # -> results_qrl_validation/selection.json (Q-DDQN)

# 3. Factorial cells of LightQDRL-X (Fig. 5)
python exp_qrl_bench.py --env mec --algos Q-DDQN --variants dqcp,jcqq --workers 4
python measure_qrl_latency.py --env mec
python exp_qrl_adapt.py --algos Q-DDQN --variants dqcp,jcqq --modes full,peqa --workers 8

# 4. Adaptation of all fourteen WP-MEC models (Fig. 6)
python exp_qrl_adapt.py --workers 8

# 5. Robustness (Fig. 7a-e)
python exp_qrl_robust.py --workers 8

# 6. Figures
python plot_qrl_fig3.py
python plot_qrl_fig4.py --env mec
python plot_qrl_fig5.py
python plot_qrl_fig6.py
python plot_qrl_fig7.py --hw_dir results_hw/Q-PPO_seed0_WK_C180_s1000 --hw_label "WK_C180, Q-PPO"
```

Approximate run time on one desktop CPU with 4 workers: WP-MEC benchmark 3–4 h, multichannel access 1 h, controls 1 h,
validation 1 h, factorial 1 h, adaptation 4–5 h, robustness 1 h. To reproduce the figures without training, download
the data deposit and place its folders as `results_qrl/`, `results_qrl_adapt/`, `results_qrl_robust/`, … (see the
README of the deposit for the mapping).

### On-device measurement (Table 3)

Copy `qrl_core.py`, `lqdrl_core.py`, `optimization.py`, `qrl_envs.py`, `measure_edge_device.py`, `data/` and
`results_qrl/mec/` (checkpoints and `reference_policies.json`) to the board, install `numpy scipy pandas psutil` and
a CPU build of PyTorch, fix the clocks (Jetson: `sudo nvpmodel -m <mode>; sudo jetson_clocks`) and run

```bash
python measure_edge_device.py --device "Jetson Orin Nano Super" --cores vqc,mlp --variants base,light --seeds 0,1,2
```

Power is read automatically from the INA3221 monitor on Jetson boards, the PMIC on Raspberry Pi 5 or RAPL on x86;
`--power manual` supports an external USB power meter.

### Hardware experiment (Fig. 7f)

The API key is read from the environment and never stored:

```bash
export ORIGIN_API_KEY=...            # PowerShell: $env:ORIGIN_API_KEY = "..."
python origin_hw_validation.py --list_backends
python origin_hw_validation.py --algo Q-PPO --seed 0 --mode origin --backend WK_C180 --shots 1000 --n_states 70 \
       --run_dir results_qrl_v1/mec/runs --env_version v1        # agent of the paper (earlier WP-MEC version)
python origin_transpile_estimate.py --fetch --backend WK_C180
python origin_transpile_estimate.py --backend WK_C180 --algo Q-PPO --seed 0 --hw_dir results_hw/Q-PPO_seed0_WK_C180_s1000
```

`--mode dryrun` runs the same protocol offline with finite-shot sampling from the exact simulator.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

The code is released under the MIT License (`LICENSE`). `optimization.py` and `data/data_10.mat` originate from the
DROO project (L. Huang, S. Bi and Y.-J. A. Zhang, *IEEE Trans. Mobile Comput.* 2020) and remain under its MIT License
(`third_party/DROO_LICENSE`).

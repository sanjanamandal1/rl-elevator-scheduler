# 🛗 RL-Based Elevator Scheduling System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?style=for-the-badge&logo=openai&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-00B140?style=for-the-badge&logo=python&logoColor=white)

**A Deep Reinforcement Learning system that learns to schedule elevators
better than classical rule-based algorithms.**

*DQN agent achieves **+9.9% improvement** over the SCAN baseline*

</div>

---

## 🎯 Project Overview

Traditional elevator scheduling uses hand-crafted rules like SCAN (the elevator
algorithm) which work well under uniform traffic but fail to adapt to:
- Dynamic traffic patterns (rush hours, off-peak times)
- Priority passengers (VIP, emergency)
- Multi-elevator coordination

This project trains RL agents to **discover scheduling policies from scratch**
purely through interaction with a simulated 20-floor building environment.

---

## 📊 Results

| Agent | Mean Reward | Std Dev | vs SCAN Baseline |
|:---|:---:|:---:|:---:|
| 🥇 **DQN (resumed)** | **+1601.65** | ±585.12 | **+9.9% ✅** |
| 🏁 SCAN Baseline | +1457.22 | ±815.02 | reference |
| Multi-Agent DQN | -963.76 | ±156.55 | training |
| Q-Learning | -1477.63 | ±110.95 | limited by table size |
| SARSA | -1493.16 | ±97.75 | limited by table size |

> **Key insight:** DQN not only scores higher (+9.9%) but is also
> **28% more stable** (±585 vs ±815) — critical for real-world deployment.

---

## 🏗️ System Architecture
```
rl_elevator/
├── env/
│   ├── building.py          # 20-floor building + priority passenger generation
│   ├── elevator.py          # Single elevator car physics model
│   └── elevator_env.py      # Gymnasium-compatible RL environment
├── agents/
│   ├── q_learning_agent.py  # Tabular Q-Learning (off-policy)
│   ├── sarsa_agent.py       # Tabular SARSA (on-policy)
│   ├── dqn_agent.py         # Deep Q-Network with replay buffer + target net
│   ├── multi_agent_dqn.py   # Independent DQN per elevator
│   └── baseline_agent.py    # Rule-based SCAN algorithm
├── training/
│   ├── train.py             # Training loops for all agents
│   └── evaluate.py          # Evaluation + plotting
├── utils/
│   └── visualize.py         # Real-time Pygame dashboard
├── config.py                # All hyperparameters
├── main.py                  # Train all agents + evaluate
├── resume.py                # Resume DQN from checkpoint
└── run_dashboard.py         # Launch live visualisation
```

---

## ✨ Features

- **4 RL agents** — Q-Learning, SARSA, DQN, Multi-Agent DQN
- **Priority passengers** — Normal (1×), VIP (2.5×), Emergency (5×) reward multipliers
- **Sinusoidal traffic patterns** — simulates morning and afternoon rush hours
- **Checkpoint resume** — continue training from saved model weights
- **Live Pygame dashboard** — real-time building visualisation with reward sparkline,
  floor queue bars, traffic pattern curve, and epsilon tracker

---

## 🚀 Setup & Installation
```bash
# 1. Clone the repository
git clone https://github.com/sanjanamandal1/rl-elevator-scheduler.git
cd rl-elevator-scheduler

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train all agents from scratch
```bash
python main.py
```

### Resume DQN from checkpoint (faster — picks up where training left off)
```bash
python resume.py
```

### Launch the live Pygame dashboard
```bash
# Watch the trained DQN agent
python run_dashboard.py dqn

# Watch Multi-Agent DQN
python run_dashboard.py madqn

# Watch SCAN baseline (compare side by side)
python run_dashboard.py scan
```

### Dashboard controls
| Key | Action |
|:---:|:---|
| `SPACE` | Toggle slow (10 FPS) / fast (60 FPS) mode |
| `R` | Reset current episode |
| `ESC` | Quit |

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Notes |
|:---|:---:|:---|
| Learning rate (α) | 0.001 | Adam optimiser |
| Discount factor (γ) | 0.95 | Values future rewards heavily |
| Epsilon start / end | 1.0 → 0.05 | ε-greedy exploration |
| Replay buffer | 20,000 | Breaks temporal correlation |
| Batch size | 128 | Stable gradient updates |
| Network | 256 → 256 → 128 | ReLU activations |
| Target net update | every 100 steps | Prevents oscillation |
| Episodes | 3,000 + 2,000 resume | Checkpoint-based |

---

## 🧠 How It Works

### Environment
The `ElevatorEnv` follows the Gymnasium interface. The **state vector** encodes
normalised floor position, direction, and passenger load per elevator, plus
up/down hall call signals per floor (46 dimensions total for 20 floors + 2 elevators).

### Reward Function
```
R(t) = Σ (delivery_bonus × priority_multiplier)
       - wait_penalty   × priority_weighted_queue
       - energy_penalty × empty_movements
```

### Why DQN beats tabular methods
Q-Learning and SARSA discretise the state space into a lookup table.
On a 20-floor building the effective state space is enormous — many states
are never visited during training. DQN uses a neural network that
**generalises across similar states**, making good decisions even on
configurations it has never seen.

---

## 📦 Requirements
```
numpy
matplotlib
pandas
gymnasium
tqdm
torch
pygame
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📁 Trained Models

After running `main.py` or `resume.py`, model weights are saved as:
- `dqn_model.pth` — DQN after initial training
- `dqn_model_resumed.pth` — DQN after checkpoint resume

> Note: `.pth` files are excluded from this repo via `.gitignore`.
> Run `python main.py` to train your own.

---

## 📄 Report

A full project report is included:
[`RL_Elevator_Scheduling_Report.docx`](./RL_Elevator_Scheduling_Report.docx)

Covers architecture, agent implementations, hyperparameter tuning,
results analysis, and conclusions.

---

## 🔮 Future Work

- [ ] PPO / A3C for more stable large-scale training
- [ ] MADDPG for explicit multi-agent communication
- [ ] Real building traffic log validation
- [ ] Continuous action spaces for speed control
- [ ] Web-based dashboard (Streamlit / Gradio)

---

<div align="center">
Made with Python · PyTorch · Gymnasium · Pygame
</div>
# ROB6323 – Unitree Go2 Reinforcement Learning Locomotion

> A reinforcement learning locomotion project for the **Unitree Go2** quadruped robot using **Isaac Lab** and **PPO**.<br>
> Trains the robot to achieve stable forward walking and accurate **lateral motion (side-stepping)**.

---

## Demo / Results

> Add your trained policy videos or result graphs here.

| Forward Walking | Side-stepping |
|:--------------:|:-------------:|
| *(video coming soon)* | *(video coming soon)* |

---

## Project Overview

| Item | Details |
|------|---------|
| Robot | Unitree Go2 (quadruped) |
| Simulator | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) |
| Algorithm | PPO ([RSL-RL](https://github.com/leggedrobotics/rsl_rl)) |
| Parallel Environments | 4,096 |
| Control Frequency | 200 Hz (decimation 4 → policy at 50 Hz) |
| Episode Length | 20 seconds |

---

## Results

The trained policy demonstrates:

- **Stable forward walking** — accurate velocity command tracking
- **Clean side-stepping** — follows lateral `vy` commands without crabbing
- **Low-torque motion** — energy-efficient joint actuation
- **Upright posture** — no crouching or knee collapse
- **Minimal foot scuffing** — clean swing phases with sufficient clearance

---

## Design

### Explicit Torque-Level PD Control

Instead of relying on the simulator's built-in PD gains, joint torques are computed explicitly:

$$\tau = K_p(q_{des} - q) - K_d\dot{q}$$

| Parameter | Value |
|-----------|-------|
| Kp | 28.0 |
| Kd | 1.0 |
| Torque limit | 23.5 Nm |

### Reward Function

| Reward Term | Role | Scale |
|-------------|------|-------|
| `track_lin_vel_xy_exp` | Forward + lateral velocity tracking | +1.25 |
| `track_lin_vel_y_exp` | Auxiliary lateral (vy) tracking | +0.50 |
| `track_ang_vel_z_exp` | Yaw rate tracking | +0.55 |
| `raibert_heuristic` | Foot placement optimization | -10.0 |
| `feet_clearance` | Foot lift during swing | -45.0 |
| `tracking_contacts_shaped_force` | Contact schedule tracking | +4.0 |
| `torque` | Torque minimization | -0.00001 |
| `orient` | Roll/pitch stability | -5.0 |
| `base_height` | Body height maintenance | -2.5 |
| `rew_action_rate` | Action smoothness | -0.0001 |

### Command Sampling Strategy

To bias training toward side-stepping, commands are sampled as follows:

```
50% — Pure side-step   (vx=0,  vy≠0)
40% — General motion   (vx, vy, ωz all random)
10% — Pure forward     (vx≠0, vy=0)
```

---

## Modified Files

Per the project rules, only the following two files were modified:

```
source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/
├── rob6323_go2_env_cfg.py   # Hyperparameters and reward scales
└── rob6323_go2_env.py       # Environment logic, rewards, and PD control
```

---

## How to Run

This project is designed to run on the **NYU Greene HPC cluster**.

### Prerequisites

- NYU Greene cluster account (`rob_gy6323-2025fa`)
- SSH access to the `burst` host
- Singularity image at `/scratch/$USER/isaac-lab-base.sif`

### Step 1: Install (first time only)

```bash
bash install.sh
```

Sets up Isaac Lab at `/scratch/$USER/IsaacLab` and prepares the Singularity image.

### Step 2: Train

```bash
bash train.sh
```

Submits a Slurm job that:
1. Runs PPO training (500 epochs, headless)
2. Automatically evaluates and renders a video after training
3. Syncs logs to the `logs/` folder

### Check Job Status

```bash
ssh burst squeue --me
```

![squeue example](docs/img/burst_squeue_example.png)

### Running Locally (requires Isaac Lab installed)

```bash
# Training
python scripts/rsl_rl/train.py --task=Template-Rob6323-Go2-Direct-v0 --headless

# Visualization
python scripts/rsl_rl/play.py --task=Template-Rob6323-Go2-Direct-v0 --checkpoint <path/to/model.pt> --video
```

---

## Project Structure

```
rob6323_go2_project/
├── source/rob6323_go2/          # Core environment code
│   └── rob6323_go2/tasks/direct/rob6323_go2/
│       ├── rob6323_go2_env.py       # Environment class
│       ├── rob6323_go2_env_cfg.py   # Configuration class
│       └── agents/
│           └── rsl_rl_ppo_cfg.py    # PPO hyperparameters
├── scripts/
│   └── rsl_rl/
│       ├── train.py             # Training script
│       └── play.py              # Evaluation / visualization script
├── docs/                        # Documentation and images
├── train.sh                     # Cluster training launcher
├── install.sh                   # Cluster environment setup
├── train.slurm                  # Slurm job script
└── tutorial.md                  # Step-by-step implementation tutorial
```

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL (PPO implementation)](https://github.com/leggedrobotics/rsl_rl)
- [Unitree Go2 Specs](https://www.unitree.com/go2/)

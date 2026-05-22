# ROB6323 – Unitree Go2 Reinforcement Learning Locomotion

> **Isaac Lab + PPO**를 이용해 Unitree Go2 4족 보행 로봇을 훈련시키는 강화학습 프로젝트입니다.  
> 안정적인 전진 보행과 **옆걸음(lateral motion)** 을 목표로 합니다.

---

## 데모 영상 / 결과

> 학습된 정책의 시연 영상이나 결과 그래프를 여기에 추가하세요.

| 전진 보행 | 옆걸음 (Side-step) |
|:---------:|:-----------------:|
| *(영상 추가 예정)* | *(영상 추가 예정)* |

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 로봇 | Unitree Go2 (4족 보행) |
| 시뮬레이터 | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) |
| 알고리즘 | PPO ([RSL-RL](https://github.com/leggedrobotics/rsl_rl)) |
| 병렬 환경 수 | 4,096개 |
| 제어 주기 | 200 Hz (decimation 4 → policy 50 Hz) |
| 에피소드 길이 | 20초 |

---

## 학습 결과

학습된 정책은 다음 능력을 보여줍니다:

- **안정적인 전진 보행** — 명령 속도를 정확하게 추종
- **옆걸음 (lateral motion)** — vy 명령에 따라 좌우 이동
- **낮은 토크 동작** — 에너지 효율적인 관절 제어
- **자세 안정성** — 무릎 꺾임, 구부정한 자세 없음
- **발 스크래핑 최소화** — swing 중 발이 지면에 끌리지 않음

---

## 핵심 설계

### 제어 방식: 명시적 토크 PD 제어

시뮬레이터 내장 PD 대신 직접 토크를 계산합니다:

$$\tau = K_p(q_{des} - q) - K_d\dot{q}$$

| 파라미터 | 값 |
|---------|-----|
| Kp | 28.0 |
| Kd | 1.0 |
| 토크 한계 | 23.5 Nm |

### 보상 함수 구성

| 보상 항목 | 역할 | 스케일 |
|----------|------|--------|
| `track_lin_vel_xy_exp` | 전진/횡이동 속도 추종 | +1.25 |
| `track_lin_vel_y_exp` | 횡이동(vy) 보조 추종 | +0.50 |
| `track_ang_vel_z_exp` | 요(yaw) 속도 추종 | +0.55 |
| `raibert_heuristic` | 발 착지 위치 최적화 | -10.0 |
| `feet_clearance` | swing 중 발 높이 확보 | -45.0 |
| `tracking_contacts_shaped_force` | 접촉 스케줄 추종 | +4.0 |
| `torque` | 토크 최소화 | -0.00001 |
| `orient` | 자세 안정화 (roll/pitch) | -5.0 |
| `base_height` | 기체 높이 유지 | -2.5 |
| `rew_action_rate` | 동작 부드러움 | -0.0001 |

### 커맨드 샘플링 전략

옆걸음에 특화된 훈련을 위해 다음 비율로 커맨드를 샘플링합니다:

```
50% — 순수 옆걸음  (vx=0,  vy≠0)
40% — 복합 이동    (vx, vy, ωz 모두 랜덤)
10% — 순수 전진    (vx≠0, vy=0)
```

---

## 수정된 파일

프로젝트 규칙에 따라 아래 두 파일만 수정했습니다:

```
source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/
├── rob6323_go2_env_cfg.py   # 하이퍼파라미터, 보상 가중치
└── rob6323_go2_env.py       # 환경 로직, 보상 계산, PD 제어
```

---

## 실행 방법

이 프로젝트는 **NYU Greene HPC 클러스터** 에서 실행하도록 설계되었습니다.

### 사전 준비

- NYU Greene 클러스터 계정 (`rob_gy6323-2025fa`)
- `burst` 호스트 SSH 접근 가능
- `/scratch/$USER/isaac-lab-base.sif` Singularity 이미지

### Step 1: 최초 설치 (1회만)

```bash
bash install.sh
```

`/scratch/$USER/IsaacLab` 에 Isaac Lab을 설치하고 Singularity 이미지를 준비합니다.

### Step 2: 학습 실행

```bash
bash train.sh
```

내부적으로 Slurm 잡을 제출하며, GPU 노드에서 다음을 순서대로 실행합니다:

1. PPO 학습 (500 epoch, headless)
2. 학습 완료 후 자동으로 `play.py`로 평가 영상 생성
3. 결과 로그를 `logs/` 폴더로 동기화

### 잡 상태 확인

```bash
ssh burst squeue --me
```

![squeue 예시](docs/img/burst_squeue_example.png)

### 로컬 환경에서 실행 (Isaac Lab 직접 설치 시)

```bash
# 학습
python scripts/rsl_rl/train.py --task=Template-Rob6323-Go2-Direct-v0 --headless

# 시각화
python scripts/rsl_rl/play.py --task=Template-Rob6323-Go2-Direct-v0 --checkpoint <path/to/model.pt> --video
```

---

## 프로젝트 구조

```
rob6323_go2_project/
├── source/rob6323_go2/          # 핵심 환경 코드
│   └── rob6323_go2/tasks/direct/rob6323_go2/
│       ├── rob6323_go2_env.py       # 환경 클래스
│       ├── rob6323_go2_env_cfg.py   # 설정 클래스
│       └── agents/
│           └── rsl_rl_ppo_cfg.py    # PPO 하이퍼파라미터
├── scripts/
│   └── rsl_rl/
│       ├── train.py             # 학습 스크립트
│       └── play.py              # 평가/시각화 스크립트
├── docs/                        # 문서 및 이미지
├── train.sh                     # 클러스터 학습 실행
├── install.sh                   # 클러스터 환경 설치
├── train.slurm                  # Slurm 잡 스크립트
└── tutorial.md                  # 구현 튜토리얼
```

---

## 참고 자료

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL (PPO 구현)](https://github.com/leggedrobotics/rsl_rl)
- [Unitree Go2 사양](https://www.unitree.com/go2/)

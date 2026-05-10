# Robot Pool Player

## Install

```bash
conda create -n mujoco_env python=3.10 -y
conda activate mujoco_env
pip install mujoco==3.2.3 numpy scipy pyyaml cvxpy osqp
```

## Run

```bash
python deploy.py
```

Optional headless run:

```bash
python deploy.py --headless
```


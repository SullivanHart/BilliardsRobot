# Robot Pool Player

<img width="1908" height="975" alt="screenshot" src="https://github.com/user-attachments/assets/b823740c-2779-46f5-b1ef-6a1c53efb0dd" />

## Overview
This is my final project for COM S 5762X: Introduction to Mobile Robotics at Iowa State. It is a controller for the Unitree G1 to play pool in a MuJoCo simulation.

https://github.com/user-attachments/assets/e6213f96-343f-4cc2-afec-fa5c6d2a69da

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


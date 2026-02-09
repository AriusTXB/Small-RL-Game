# 2-Agent Shooter RL Game

A reinforcement learning environment where two agents compete in a maze. The agents learn to navigate, shoot, and avoid enemy fire using PPO (Proximal Policy Optimization).

## Features

- Custom Gymnasium Environment: ShooterEnv with procedurally generated mazes.

- Self-Play Training: Agents train against a copy of themselves using a shared PPO policy.

- Real-Time Visualization: Built-in Pygame rendering to watch the agents learn.

- Manual Control: Take control of the Green agent using Arrow Keys and Spacebar after 1000 steps.

## Requirements

- Python 3.10+

- gymnasium

- stable-baselines3

- torch

- pygame

- numpy

- fastapi

- uvicorn

## Setup & Run

- Create Virtual Environment:

'code

Bash

download

content_copy

expand_less

cd backend

python -m venv venv'
## Windows

.\venv\Scripts\activate

## Linux/Mac

source venv/bin/activate

## Install Dependencies:

- code
- Bash
- download
- content_copy
- expand_less
- pip install -r requirements.txt

## Run Training:

- code
- Bash
- download
- content_copy
- expand_less
- python main.py

The game window will open automatically.

## Project Structure
- code
- Text
- download
- content_copy
- expand_less
backend/

├── main.py        # Entry point for training and game loop

├── env/           # Custom RL environment code (ShooterEnv)

├── models/        # Saved model checkpoints

└── docs/          # Project documentation and specifications

## Manual control is available for the Green agent:
- Arrow Keys: Move

- Space: Shoot


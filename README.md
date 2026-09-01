# DQN Snake Agent with a Custom Magnet Mechanic

A Deep Reinforcement Learning project in which a Deep Q-Network learns to play a custom Snake environment implemented from scratch in Python.

I developed this academic project at Aristotle University of Thessaloniki as part of the MSc program in Digital Media - Computational Intelligence.

## Overview

The agent learns to collect food, avoid walls and self-collisions, and maximize its long-term reward. The environment also includes an optional **magnet mechanic**: after activation, nearby food can be collected from a distance of one grid cell, adding an extra strategic element to the classic game.

## Reinforcement-learning approach

The implementation uses a Deep Q-Network with:

- A 16-feature state representation
- Three relative actions: turn left, move straight, or turn right
- Two fully connected hidden layers with 256 units each
- Experience replay with a capacity of 10,000 transitions
- A separate target network for stable learning
- Epsilon-greedy exploration with gradual decay
- Adam optimization and mean squared error loss
- Discount factor `gamma = 0.99`

## Environment and rewards

The custom `SnakeEnv` uses an 8 x 8 grid by default. The state includes collision risks, movement direction, relative food position, normalized coordinates, and snake occupancy.

The reward design includes:

- **+10** for collecting food
- **-10** for a collision
- A small step penalty
- Additional shaping based on whether the snake moves closer to the food

## Results

According to the accompanying report:

| Metric | Result |
|---|---:|
| Final average training reward | **83.89** |
| Evaluation games | 5 |
| Average evaluation score | **20.20** |
| Observed evaluation score range | 15-27 |

The trained policy showed consistent food collection and obstacle avoidance without random exploration during evaluation.

## Running the notebook

1. Create and activate a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Open `RL.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells to define the environment, model, replay buffer, and training functions.
5. Train the agent before evaluation so that `best_snake_magnet_dqn.pth` is created.

Training 1,000 episodes can be started through the notebook's main execution section. Rendering is best enabled during evaluation rather than training.

## Repository files

- `RL.ipynb` - complete environment, DQN training, evaluation, and visualization code
- `RL.pdf` - original Greek academic report with methodology and results
- `requirements.txt` - Python dependencies

## Limitations

The environment follows an educational Gym-style interface but does not inherit directly from Gym or Gymnasium. Results can vary between runs because the notebook does not set random seeds. The saved model checkpoint is generated locally and is excluded from version control.

## Author

**Georgios Kolokotronis**  
MSc in Digital Media - Computational Intelligence  
Aristotle University of Thessaloniki

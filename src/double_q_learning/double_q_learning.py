from collections import deque
import random
import numpy as np
from tqdm import tqdm
from src.config import *
from src.utils import get_defender_action

def choose_action(Q1, Q2, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(n_actions))
    else:
        Q_combined = Q1 + Q2
        return np.argmax(Q_combined)

def update_Q(Q1, Q2, state, action, reward, next_state, alpha, gamma, update_Q1=True):
    if update_Q1:
        # Update Q1 using Q2 for the next state
        best_action = np.argmax(Q2)
        Q1[action] += alpha * (reward + gamma * Q2[best_action] - Q1[action])
    else:
        # Update Q2 using Q1 for the next state
        best_action = np.argmax(Q1)
        Q2[action] += alpha * (reward + gamma * Q1[best_action] - Q2[action])

def train_double_q_learning():
    Q1 = np.zeros(n_actions)  # Initialize Q1 table for a single state
    Q2 = np.zeros(n_actions)  # Initialize Q2 table for a single state
    q_value_history = []  # To store the history of Q-values
    epsilon = initial_epsilon  # Initial epsilon for epsilon-greedy policy

    print("Starting training...")
    for episode in tqdm(range(episodes), desc="Training Progress"):
        done = False  # Episode completion flag
        episode_rewards = []  # To store rewards for the current episode

        while not done:
            # Choose action using epsilon-greedy policy
            attacker_action = choose_action(Q1, Q2, 0, epsilon)  # Single state is 0
            defender_action = get_defender_action(actions)  # Get defender's action
            attacker_number = actions[attacker_action]
            defender_number = actions[defender_action]

            if attacker_number != defender_number:
                reward = attacker_number  # Attacker wins, gets points
            else:
                reward = 0  # Defender wins, no points for attacker
                done = True  # End the episode

            # Update Q-values using Double Q-learning update rule
            update_Q(Q1, Q2, 0, attacker_action, reward, 0, alpha, gamma, update_Q1=random.uniform(0, 1) < 0.5)

            episode_rewards.append(reward)  # Collect reward

        # Store the Q-values for each action after every 100 episodes
        if (episode + 1) % 100 == 0:
            Q_combined = Q1 + Q2
            q_value_history.append(Q_combined.copy())

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Periodically print progress
        if (episode + 1) % 10000 == 0:
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Epsilon: {epsilon:.6f}")
            for i, action in enumerate(actions):
                q_value = Q_combined[i]
                print(f"Action: {action:3d}, Q-value: {q_value:.4f}")
            print("\n")

    return Q1, Q2, q_value_history
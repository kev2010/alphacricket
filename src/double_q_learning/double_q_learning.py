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
        Q_combined = Q1[state, :] + Q2[state, :]
        return np.argmax(Q_combined)

def update_Q(Q1, Q2, state, action, reward, next_state, alpha, gamma, update_Q1=True):
    if update_Q1:
        best_action = np.argmax(Q1[next_state, :])
        Q1[state, action] += alpha * (reward + gamma * Q2[next_state, best_action] - Q1[state, action])
    else:
        best_action = np.argmax(Q2[next_state, :])
        Q2[state, action] += alpha * (reward + gamma * Q1[next_state, best_action] - Q2[state, action])

def train_double_q_learning():
    Q1 = np.zeros((max_points + 1, n_actions))  # Initialize Q1 table
    Q2 = np.zeros((max_points + 1, n_actions))  # Initialize Q2 table
    q_value_history = []  # To store the history of Q-values
    epsilon = initial_epsilon  # Initial epsilon for epsilon-greedy policy

    # Experience replay buffer
    replay_buffer = deque(maxlen=10000)
    batch_size = 64

    # Moving average parameters for reward smoothing
    reward_window = 100
    reward_smoothing = deque(maxlen=reward_window)

    print("Starting training...")
    for episode in tqdm(range(episodes), desc="Training Progress"):
        state = 0  # Initial state
        done = False  # Episode completion flag
        episode_rewards = []  # To store rewards for the current episode

        while not done:
            # Choose action using epsilon-greedy policy
            attacker_action = choose_action(Q1, Q2, state, epsilon)
            defender_action = get_defender_action(actions)  # Get defender's action
            attacker_number = actions[attacker_action]
            defender_number = actions[defender_action]

            if attacker_number != defender_number:
                reward = attacker_number  # Attacker wins, gets points
                next_state = min(state + reward, max_points)  # Update state
            else:
                reward = 0  # Defender wins, no points for attacker
                next_state = state  # State remains the same
                done = True  # End the episode

            # Store experience in replay buffer
            replay_buffer.append((state, attacker_action, reward, next_state, done))

            # Sample a batch of experiences from the replay buffer
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                for b_state, b_action, b_reward, b_next_state, b_done in batch:
                    # Update Q-values using Double Q-learning update rule
                    update_Q(Q1, Q2, b_state, b_action, b_reward, b_next_state, alpha, gamma, update_Q1=random.uniform(0, 1) < 0.5)

            state = next_state  # Move to the next state
            episode_rewards.append(reward)  # Collect reward

            if state == max_points:
                done = True  # End the episode if max points are reached

        # Smooth the rewards using a moving average
        reward_smoothing.append(np.sum(episode_rewards))
        smoothed_reward = np.mean(reward_smoothing)
        q_value_history.append(smoothed_reward)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Periodically print progress
        if (episode + 1) % 10000 == 0:
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Epsilon: {epsilon:.6f}")
            Q_combined = Q1[0, :] + Q2[0, :]
            for i, action in enumerate(actions):
                q_value = Q_combined[i]
                print(f"Action: {action:3d}, Q-value: {q_value:.4f}")
            print("\n")

    return Q1, Q2, q_value_history
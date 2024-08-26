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
    Q1 = np.zeros((max_points + 1, n_actions))
    Q2 = np.zeros((max_points + 1, n_actions))
    q_value_history = []
    epsilon = initial_epsilon

    print("Starting training...")
    for episode in tqdm(range(episodes), desc="Training Progress"):
        state = 0
        done = False

        while not done:
            attacker_action = choose_action(Q1, Q2, state, epsilon)
            defender_action = get_defender_action(actions)
            attacker_number = actions[attacker_action]
            defender_number = actions[defender_action]

            if attacker_number != defender_number:
                reward = attacker_number
                next_state = min(state + reward, max_points)
            else:
                reward = 0
                next_state = state
                done = True

            if random.uniform(0, 1) < 0.5:
                update_Q(Q1, Q2, state, attacker_action, reward, next_state, alpha, gamma, update_Q1=True)
            else:
                update_Q(Q1, Q2, state, attacker_action, reward, next_state, alpha, gamma, update_Q1=False)

            state = next_state

            if state == max_points:
                done = True

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % 100 == 0:
            Q_combined = Q1[0, :] + Q2[0, :]
            q_value_history.append(Q_combined.copy())

        if (episode + 1) % 10000 == 0:
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Epsilon: {epsilon:.6f}")
            Q_combined = Q1[0, :] + Q2[0, :]
            for i, action in enumerate(actions):
                q_value = Q_combined[i]
                print(f"Action: {action:3d}, Q-value: {q_value:.4f}")
            print("\n")

    return Q1, Q2, q_value_history
import numpy as np
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define possible actions (numbers that can be chosen)
actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 90, 100]
n_actions = len(actions)

# Cap for accumulated points
max_points = 1000

# Initialize Q-tables (2D arrays: state x action)
Q1 = np.zeros((max_points + 1, n_actions))
Q2 = np.zeros((max_points + 1, n_actions))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
# initial_temperature = 10.0  # Start with a higher temperature
# min_temperature = 0.1  # Minimum temperature
# temperature_decay = 0.9999  # Slower decay rate
episodes = 100000000  # Number of training episodes
initial_epsilon = 1.0  # Start with full exploration
min_epsilon = 0.01  # Minimum exploration rate
epsilon_decay = 0.99999995  # Decay rate for 100 million episodes

# def boltzmann_exploration(Q1, Q2, state, temperature):
#     # Combine Q-values from both tables
#     Q_combined = Q1[state, :] + Q2[state, :]
#     
#     # Apply temperature scaling
#     scaled_Q = Q_combined / temperature
#     
#     # Subtract max value for numerical stability
#     scaled_Q -= np.max(scaled_Q)
#     
#     # Calculate probabilities
#     exp_Q = np.exp(scaled_Q)
#     probabilities = exp_Q / np.sum(exp_Q)
#     
#     # Handle any remaining numerical instability
#     probabilities = np.nan_to_num(probabilities, nan=1.0/len(actions))
#     probabilities /= np.sum(probabilities)
#     
#     # Choose action based on probabilities
#     return np.random.choice(len(actions), p=probabilities)
    
def choose_action(Q1, Q2, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        return random.choice(range(n_actions))
    else:
        # Exploit: choose the best action from the combined Q-values
        Q_combined = Q1[state, :] + Q2[state, :]
        return np.argmax(Q_combined)

def get_defender_action():
    # Defender plays based on the weights of actions
    total_weight = sum(actions)
    probabilities = [action / total_weight for action in actions]
    # Return the index of the chosen action as a single number
    return np.random.choice(range(n_actions), p=probabilities)

def update_Q(Q1, Q2, state, action, reward, next_state, alpha, gamma, update_Q1=True):
    if update_Q1:
        best_action = np.argmax(Q1[next_state, :])
        Q1[state, action] += alpha * (reward + gamma * Q2[next_state, best_action] - Q1[state, action])
    else:
        best_action = np.argmax(Q2[next_state, :])
        Q2[state, action] += alpha * (reward + gamma * Q1[next_state, best_action] - Q2[state, action])

# Initialize a list to store Q-values over time
q_value_history = []

print("Starting training...")
epsilon = initial_epsilon
# temperature = initial_temperature  # Initialize temperature
for episode in tqdm(range(episodes), desc="Training Progress"):
    # Start from an initial state (e.g., 0 points)
    state = 0
    done = False
    
    while not done:
        # Attacker and defender choose actions
        # attacker_action = boltzmann_exploration(Q1, Q2, state, temperature)
        attacker_action = choose_action(Q1, Q2, state, epsilon)
        defender_action = get_defender_action()
        
        # Convert actions back to actual numbers
        attacker_number = actions[attacker_action]
        defender_number = actions[defender_action]
        
        # Determine reward and next state
        if attacker_number != defender_number:
            reward = attacker_number  # Attacker gets points if numbers don't match
            next_state = min(state + reward, max_points)  # Cap the points at max_points
        else:
            reward = 0  # No points if numbers match
            next_state = state  # State remains the same
            done = True

        # Randomly choose which Q-function to update
        if random.uniform(0, 1) < 0.5:
            update_Q(Q1, Q2, state, attacker_action, reward, next_state, alpha, gamma, update_Q1=True)
        else:
            update_Q(Q1, Q2, state, attacker_action, reward, next_state, alpha, gamma, update_Q1=False)

        state = next_state

        if state == max_points:
            done = True  # End episode when max points reached

    # In the training loop, adjust the temperature decay
    # temperature = max(min_temperature, initial_temperature * (temperature_decay ** episode))

    # Decrease epsilon over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Store Q-values for initial state every 100 episodes
    if episode % 100 == 0:
        Q_combined = Q1[0, :] + Q2[0, :]
        q_value_history.append(Q_combined.copy())

 # Print update every 10000 episodes
    if (episode + 1) % 10000 == 0:
        print(f"\nEpisode {episode + 1}/{episodes}")
        # print(f"Temperature: {temperature:.4f}")
        print(f"Epsilon: {epsilon:.6f}")
        
        # Print Q-values for all actions from the initial state
        Q_combined = Q1[0, :] + Q2[0, :]
        print("Q-values for all actions from the initial state:")
        for i, action in enumerate(actions):
            q_value = Q_combined[i]
            print(f"Action: {action:3d}, Q-value: {q_value:.4f}")
        
        print("\n")  # Add some space for readability

# After training, plot Q-value history
plt.figure(figsize=(14, 10))
q_value_history = np.array(q_value_history)
x_axis = np.arange(0, episodes, 100)  # Adjust this to match your storage frequency

# Use a color map that ensures distinct colors
colors = plt.cm.rainbow(np.linspace(0, 1, len(actions)))

for i, action in enumerate(actions):
    plt.plot(x_axis, q_value_history[:, i], label=f'Action {action}', color=colors[i])

plt.xlabel('Episode')
plt.ylabel('Q-value')
plt.title('Q-values for Initial State Over Time')

# Improve legend placement and readability
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
plt.tight_layout()

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate the final Q-values
for i, action in enumerate(actions):
    final_q = q_value_history[-1, i]
    plt.annotate(f'{action}: {final_q:.2f}', 
                 xy=(x_axis[-1], final_q), 
                 xytext=(5, 0), 
                 textcoords='offset points', 
                 ha='left', 
                 va='center',
                 fontsize=8,
                 color=colors[i])

plt.savefig('q_value_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Q-value history plot saved as q_value_history.png")

# After the training loop
print("Training complete. Analyzing results...")

# Combine Q-values from both tables for the initial state
Q_combined = Q1[0, :] + Q2[0, :]

print("\nQ-values for all actions from the initial state (0 points):")
for i, action in enumerate(actions):
    q_value = Q_combined[i]
    print(f"Action: {action:3d}, Q-value: {q_value:.2f}")

# Optional: Sort actions by Q-value (descending order)
sorted_actions = sorted(zip(actions, Q_combined), key=lambda x: x[1], reverse=True)
print("\nActions sorted by Q-value (descending order):")
for action, q_value in sorted_actions:
    print(f"Action: {action:3d}, Q-value: {q_value:.2f}")

# After the training loop and result analysis
print("Saving Q-values to file...")

# Combine Q-values from both tables
Q_combined = Q1 + Q2

# Create a dictionary to store Q-values for each state
Q_values_dict = {}
for state in range(max_points + 1):
    Q_values_dict[state] = {action: float(Q_combined[state, i]) for i, action in enumerate(actions)}

# Save Q-values to a JSON file
with open('q_values.json', 'w') as f:
    json.dump(Q_values_dict, f, indent=2)

print("Q-values saved to q_values.json")
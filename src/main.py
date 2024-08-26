import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.double_q_learning.double_q_learning import train_double_q_learning
from src.config import actions, max_points
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    # Train Q-learning model
    Q1, Q2, q_value_history = train_double_q_learning()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Convert q_value_history to a NumPy array for easier indexing
    q_value_history = np.array(q_value_history)

    # Ensure q_value_history is 2-dimensional
    if q_value_history.ndim == 1:
        if q_value_history.size % len(actions) == 0:
            q_value_history = q_value_history.reshape(-1, len(actions))
        else:
            raise ValueError(f"Cannot reshape array of size {q_value_history.size} into shape (-1, {len(actions)})")

    # Plot Q-values for each action over time
    plt.figure(figsize=(14, 10))
    for i, action in enumerate(actions):
        plt.plot(q_value_history[:, i], label=f'Action {action}')

    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.title('Q-values Over Time for Each Action')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save plot to results folder
    plt.savefig('results/q_values_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Q-values plot saved as results/q_values_over_time.png")

    # Combine Q1 and Q2 and save to JSON
    Q_combined = Q1 + Q2
    Q_values_dict = {action: float(Q_combined[i]) for i, action in enumerate(actions)}

    with open('results/q_values.json', 'w') as f:
        json.dump(Q_values_dict, f, indent=2)

    print("Q-values saved to results/q_values.json")

if __name__ == "__main__":
    main()
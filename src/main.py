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

    # Plot Q-value history
    plt.figure(figsize=(14, 10))
    q_value_history = np.array(q_value_history)
    x_axis = np.arange(0, len(q_value_history) * 100, 100)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(actions)))

    for i, action in enumerate(actions):
        plt.plot(x_axis, q_value_history[:, i], label=f'Action {action}', color=colors[i])

    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.title('Q-values for Initial State Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)

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

    # Save plot to results folder
    plt.savefig('results/q_value_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Q-value history plot saved as results/q_value_history.png")

    # Combine Q1 and Q2 and save to JSON
    Q_combined = Q1 + Q2
    Q_values_dict = {state: {action: float(Q_combined[state, i]) for i, action in enumerate(actions)} for state in range(max_points + 1)}

    with open('results/q_values.json', 'w') as f:
        json.dump(Q_values_dict, f, indent=2)

    print("Q-values saved to results/q_values.json")

if __name__ == "__main__":
    main()
import numpy as np

q_values = [1062.49, 921.62, 917.01, 914.47, 913.95, 913.70, 906.75, 898.18, 890.23, 889.62, 888.57, 855.25, 851.41]
actions = [100, 90, 4, 7, 9, 3, 6, 5, 8, 2, 50, 10, 1]

# Normalize q_values so they sum to 10
q_values_normalized = np.array(q_values) * (10 / np.sum(q_values))

# Calculate softmax probabilities
exp_q = np.exp(q_values_normalized)
softmax_probs = exp_q / np.sum(exp_q)

# Print results
for action, prob in zip(actions, softmax_probs):
    print(f"Action: {action:3d}, Probability: {prob:.6f}")
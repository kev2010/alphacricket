import nashpy as nash
import numpy as np
from scipy.optimize import linprog

# Define the actions
actions = [1, 5, 10, 50, 90, 100]

# Create the payoff matrix for the attacker
payoff_matrix = np.zeros((len(actions), len(actions)))

# Populate the payoff matrix
for i, attacker_action in enumerate(actions):
    for j, defender_action in enumerate(actions):
        if attacker_action != defender_action:
            # Attacker wins and accumulates points
            payoff_matrix[i, j] = attacker_action
        else:
            # Defender wins, attacker gets 0 points
            payoff_matrix[i, j] = 0

# Create the game
game = nash.Game(payoff_matrix)

# Find the Nash Equilibrium using vertex enumeration
equilibria = list(game.vertex_enumeration())

# Print the equilibria
for eq in equilibria:
    print(f"Attacker strategy: {eq[0]}")
    print(f"Defender strategy: {eq[1]}")
    print()

# Calculate and print the expected value of each Nash equilibrium strategy
for eq in equilibria:
    attacker_strategy, defender_strategy = eq
    expected_value_nash = 0
    for i, attacker_action in enumerate(actions):
        for j, defender_action in enumerate(actions):
            expected_value_nash += attacker_strategy[i] * defender_strategy[j] * payoff_matrix[i, j]
    print(f"Expected value of Nash equilibrium strategy: {expected_value_nash}")

# Define a fixed strategy for the defender
# Define a weighted average strategy for the defender based on the point values in actions
total_points = sum(actions)
fixed_defender_strategy = [action / total_points for action in actions]

# Find the optimal attacker strategy to exploit the fixed defender strategy
c = -np.dot(payoff_matrix, fixed_defender_strategy)  # Coefficients for the objective function (negative for maximization)
A_eq = np.ones((1, len(actions)))  # Sum of probabilities must be 1
b_eq = np.array([1])
bounds = [(0, 1) for _ in actions]  # Probabilities must be between 0 and 1

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if result.success:
    optimal_attacker_strategy = result.x
    print(f"Optimal attacker strategy to exploit fixed defender strategy: {optimal_attacker_strategy}")
else:
    print("Failed to find the optimal attacker strategy.")
    optimal_attacker_strategy = None

# Calculate the expected value for the attacker using the optimal strategy
if optimal_attacker_strategy is not None:
    expected_value = 0
    for i, attacker_action in enumerate(actions):
        for j, defender_action in enumerate(actions):
            expected_value += optimal_attacker_strategy[i] * fixed_defender_strategy[j] * payoff_matrix[i, j]

    print(f"Expected value for the attacker against fixed defender strategy: {expected_value}")
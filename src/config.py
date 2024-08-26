# Configuration parameters
# actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 90, 100]
actions = [1, 5, 10]
n_actions = len(actions)
max_points = 1000
alpha = 0.1
gamma = 0.9
episodes = 1_000_000
initial_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999995
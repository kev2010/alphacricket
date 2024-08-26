import numpy as np
import random

def get_defender_action(actions):
    total_weight = sum(actions)
    probabilities = [action / total_weight for action in actions]
    return np.random.choice(range(len(actions)), p=probabilities)
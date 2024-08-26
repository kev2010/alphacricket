# alphacricket

## Game

"Hand Cricket" is a game where two players choose a number from the set {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 90, 100} simultaneously. One player acts as the "attacker" aiming to accumulate points, while the other is the "defender" attempting to prevent the attacker from scoring.

If the attacker selects a number different from the defender's choice, the attacker scores that number of points. If both players choose the same number, the defender wins the round, and the attacker's score is finalized. The roles then switch, and the player with the highest score at the end wins the game.

Here, I'm just running some different reinforcement learning experiments on this game :)

## Setup

1. Create a virtual environment:

   ```
   python3 -m venv alphacricket_env
   ```

2. Activate the virtual environment:

   ```
   source alphacricket_env/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. To run:

   ```
   python3 src/main.py
   ```

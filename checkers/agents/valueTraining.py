from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

import sys
import time
import os

# Add the path to the directory containing the checkers package
sys.path.append(os.path.abspath('/Users/ethanford/School/cs5060/cs5060final'))
from checkers.game import Checkers
from checkers.agents.baselines import RandomPlayer
from checkers.agents.alpha_beta import MinimaxPlayer

# Define the model for the value function
value_model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    solver='adam',
    warm_start=False,
    max_iter=100,
    tol=0.00001,
    verbose=True,
    alpha=0.0001
)

# Prepare initial dummy training data
initial_X = np.random.rand(10, 32)  # 32 playable squares
initial_y = np.random.rand(10)      # Random target values
value_model.fit(initial_X, initial_y)  # Initial training

# Self-play function to generate training data
def generate_training_data(num_games=100):
    X = []
    y = []
    for _ in range(num_games):
        game = Checkers()
        history = []
        winner = None

        while winner is None:
            legal_moves = game.legal_moves()
            if not legal_moves:
                winner = game.adversary 
                break

            move = legal_moves[0]
            board, turn, last_moved_piece, next_moves, winner = game.move(*move)

            # Convert the board state to a 32-length numeric array
            state = np.zeros(32, dtype=int)
            for sq in game.board['black']['men']:
                state[sq] = 1  # Black men
            for sq in game.board['black']['kings']:
                state[sq] = 2  # Black kings
            for sq in game.board['white']['men']:
                state[sq] = 3  # White men
            for sq in game.board['white']['kings']:
                state[sq] = 4  # White kings

            history.append((state, turn))

        result = 1 if winner == 'black' else -1 if winner == 'white' else 0
        for state, turn in history:
            X.append(state)
            y.append(result if turn == 'black' else -result)

    return np.array(X), np.array(y)

# Incremental training
def incremental_training(value_model, iterations=10, games_per_iter=100):
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        X, y = generate_training_data(num_games=games_per_iter)
        value_model.partial_fit(X, y)  # Train incrementally




def generate_next_states(state):
    """
    Mock function to generate possible next states.
    Replace with game-specific logic to simulate legal moves.

    Args:
        state: The current state as a numeric array.

    Returns:
        A list of next possible states.
    """
    rng = np.random.default_rng()
    return [state + rng.uniform(-0.1, 0.1, state.shape) for _ in range(5)]  # Example: 5 possible states


def evaluate_state_with_timer(value_model, state, depth, start_time, time_limit):
    """
    Recursively evaluate a state with a time constraint.
    
    Args:
        value_model: Trained MLPRegressor value function.
        state: The current state as a numeric array.
        depth: Remaining search depth.
        start_time: Time when evaluation started.
        time_limit: Maximum allowed time in seconds.
        threshold: Stopping threshold for reward improvement.

    Returns:
        Predicted reward for the given state and depth.
    """
    elapsed_time = time.time() - start_time
    if elapsed_time >= time_limit or depth == 0:
        # Base case: Evaluate state directly if depth is 0 or time is up
        return value_model.predict(state.reshape(1, -1))[0]

    # Generate possible next states (replace with game-specific logic)
    next_states = generate_next_states(state)

    # Recursively evaluate the next states
    max_next_reward = max(
        evaluate_state_with_timer(value_model, next_state, depth - 1, start_time, time_limit)
        for next_state in next_states
    )

    # Stop if improvement is below the threshold
    return max_next_reward

# Modified find_optimal_depth to incorporate dynamic depth adjustment
def find_optimal_depth_with_timer(value_model, max_depth, num_samples=100, time_limit=40):
    """
    Find the optimal search depth using a timer.

    Args:
        value_model: Trained MLPRegressor value function.
        max_depth: Maximum depth to evaluate.
        num_samples: Number of random states to evaluate per depth.
        time_limit: Maximum allowed time in seconds.

    Returns:
        The optimal search depth.
    """
   # Initialize data to track search depth over time
    time_per_depth = []
    depths = []

    rewards_by_depth = {}

    # Start timer
    start_time = time.time()

    # Randomly generate sample states for evaluation
    rng = np.random.default_rng()
    states = [rng.uniform(0, 1, 32) for _ in range(num_samples)]  # 32 squares in Checkers

    for depth in range(1, max_depth + 1):
        # Check if the time limit is exceeded
        if time.time() - start_time >= time_limit:
            print(f"Time limit exceeded. Stopping at depth {depth-1}.")
            break

        # Evaluate total reward for this depth
        total_reward = 0
        for state in states:
            total_reward += evaluate_state_with_timer(value_model, state, depth, start_time, time_limit)
            rewards_by_depth[depth] = total_reward

        # Record the elapsed time and depth
        elapsed_time = time.time() - start_time
        time_per_depth.append(elapsed_time)
        depths.append(depth)

    print(time_per_depth)

    # Plotting Search Depth vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(time_per_depth, depths, marker='o', linestyle='-', linewidth=2)
    plt.title("Search Depth vs Elapsed Time", fontsize=16)
    plt.xlabel("Elapsed Time (seconds)", fontsize=14)
    plt.ylabel("Search Depth", fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()





    # Find the depth with the highest average reward
    optimal_depth = max(rewards_by_depth, key=rewards_by_depth.get)
    print("Rewards by Depth:", rewards_by_depth)
    print(f"Optimal Depth: {optimal_depth}")
    return optimal_depth


    # Simulate solution quality vs search depth
def solution_quality_vs_depth(value_model, game_state, max_depth):
    """
    Evaluate solution quality at different search depths.
    
    Args:
        value_model: Trained MLPRegressor value function.
        game_state: An arbitrary game state (e.g., board array).
        max_depth: Maximum depth to evaluate.

    Returns:
        A list of solution qualities at each depth.
    """
    solution_qualities = []
    best_solution = None

    for depth in range(1, max_depth + 1):
        # Evaluate the state at the current depth
        start_time = time.time()
        quality = evaluate_state_with_timer(value_model, game_state, depth, start_time, time_limit=10)

        # Store the quality at this depth
        solution_qualities.append(quality)

        # Update the best solution if at max depth
        if depth == max_depth:
            best_solution = quality

    # Normalize qualities relative to the maximum depth solution
    normalized_qualities = [q / best_solution for q in solution_qualities]

    return normalized_qualities


# Arbitrary game state (mock)
game = Checkers()

state = np.zeros(32, dtype=int)
for sq in game.board['black']['men']:
    state[sq] = 1  # Black men
for sq in game.board['black']['kings']:
    state[sq] = 2  # Black kings
for sq in game.board['white']['men']:
    state[sq] = 3  # White men
for sq in game.board['white']['kings']:
    state[sq] = 4  # White kings
 


# Run the analysis
max_depth = 10
qualities = solution_quality_vs_depth(value_model, state, max_depth)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_depth + 1), qualities, marker='o', linestyle='-', linewidth=2)
plt.title("Solution Quality vs Search Depth", fontsize=16)
plt.xlabel("Search Depth", fontsize=14)
plt.ylabel("Normalized Solution Quality", fontsize=14)
plt.grid(True)
plt.xticks(range(1, max_depth + 1), fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Run training
if __name__ == "__main__":

    print("\n-----OPTIMAL DEPTH-----")
    optimal_depth = find_optimal_depth_with_timer(value_model, max_depth=10, num_samples=100)
    print(optimal_depth)

    print("\n TRAINING")

    model = incremental_training(value_model)


    player1 = MinimaxPlayer('black', value_func=model, search_depth=optimal_depth)
    player2 = MinimaxPlayer('white', value_func=model, search_depth=optimal_depth)

    # # Play a game
    # game = Checkers()
    # winner = None
    # while winner is None:
    #     if game.turn == 'black':
    #         move = player1.next_move(game.board, game.last_moved_piece)
    #     else:
    #         move = player2.next_move(game.board, game.last_moved_piece)

    #     game.move(*move)
    #     winner = game.adversary if not game.legal_moves() else None

    # print(f"Winner: {winner}")

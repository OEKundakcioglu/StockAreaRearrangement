import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from Env_wnr import CustomEnv, render
import Env as Env
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def preprocess_state_with_priority(state, remaining_moves, moves=3):
    processed_state = []
    for position_key in sorted(state.keys()):
        bobbin = state[position_key]
        if bobbin is None:
            processed_state.extend([0, 0, 0, 0, 0, 0])
        else:
            priority_encoding = [0, 0, 0, 0, 0]
            priority_encoding[bobbin.priority - 1] = 1
            is_blocked_o = 1 if bobbin.is_blocked_o else 0
            processed_state.extend(priority_encoding + [is_blocked_o])
    moves_normalized = (remaining_moves - 1) / (
        moves - 1
    )  # Normalizing based on max moves (3)
    processed_state.append(moves_normalized)
    return np.array(processed_state).reshape(1, -1)


def get_position_coordinates(position_key):
    """Get the row and column coordinates from the position key."""
    row, col = map(int, position_key.split("_")[1].split(","))
    return int(row), int(col)


def solve_with_model(model, state_initial, env):
    # Add your code here
    remaining_moves = 3
    state_count = 0

    env.state = state_initial.copy()
    state = state_initial.copy()

    states = {}

    moves = {}

    state = preprocess_state_with_priority(state, remaining_moves)

    done = False
    total_reward = 0

    def get_valid_actions():
        valid_actions = []
        valid_actions.append(3720)  # Add action number 3720 for "do nothing"
        for action in range(0, 3720):
            from_key, to_key = env.action_to_key(action)

            # Ensure both keys are valid before checking move validity
            if from_key is None or to_key is None:
                continue
            if env.is_move_valid(from_key, to_key):
                valid_actions.append(action)

        return valid_actions

    while not done:

        # Predict the best action from the current state
        q_values = model.predict(state)
        valid_actions = get_valid_actions()
        q_values_filtered = q_values[0][valid_actions]
        action = valid_actions[np.argmax(q_values_filtered)]

        if env.perfect_scenerio():
            done = True
            print("Perfect Scenerio")
        render(env.state)
        if done == False:
            # Execute the chosen action
            next_state, reward, done, _ = env.step(action, total_reward)

            state_count += 1
            remaining_moves -= 1

            # f"state{statecount}" = nextstate
            states[f"state_{state_count}"] = next_state.copy()
            moves[f"move_{state_count}"] = env.action_to_key(action)

            next_state = preprocess_state_with_priority(next_state, remaining_moves)

            # Update state and total reward
            state = next_state
            total_reward += reward

            # Render the environment (optional)

    return states, moves, total_reward


def model_use(env, state_initial, max_reward):

    model_names = [
        "model1.keras",
        "model2.keras",
        "model3.keras",
        "model4.keras",
        # "model5.keras",
        # "model6.keras",
        # "model7.keras",
        # "model8.keras",
        "model9.keras",
        "model10.keras",
        # "model11.keras",
        "model12.keras",
        "model13.keras",
        "model14.keras",
        "model15.keras",
        "model16.keras",
    ]
    states = {}
    moves = {}
    best_states = {}
    best_moves = {}
    best_model = None

    for model_name in model_names:
        env.state = state_initial.copy()
        env.total_movements = 0
        states, moves, total_reward = solve_with_model(
            load_model(model_name), state_initial.copy(), env
        )
        if total_reward > 200 and env.total_movements <= 1:
            total_reward += 200
        elif total_reward > 200 and env.total_movements <= 2:
            total_reward += 100  # these aditio
        if total_reward > max_reward:
            best_model = model_name
            max_reward = total_reward
            best_states = states
            best_moves = moves

    if max_reward >= 400:
        max_reward -= 200
    elif max_reward >= 300:
        max_reward -= 100

    return max_reward, best_states, best_moves, best_model


def main():
    env = CustomEnv()
    # Reset the environment and preprocess the initial state
    # Load the trained model
    episodes = 0

    model_names = {
        "model1.keras": 0,
        "model2.keras": 0,
        "model3.keras": 0,
        "model4.keras": 0,
        # "model5.keras": 0,
        # "model6.keras": 0,
        # "model7.keras": 0,
        # "model8.keras": 0,
        "model9.keras": 0,
        "model10.keras": 0,
        # "model11.keras": 0,
        "model12.keras": 0,
        "model13.keras": 0,
        "model14.keras": 0,
        "model15.keras": 0,
        "model16.keras": 0,
    }
    rewards = {
        "model1.keras": 0,
        "model2.keras": 0,
        "model3.keras": 0,
        "model4.keras": 0,
        # "model5.keras": 0,
        # "model6.keras": 0,
        # "model7.keras": 0,
        # "model8.keras": 0,
        "model9.keras": 0,
        "model10.keras": 0,
        # "model11.keras": 0,
        "model12.keras": 0,
        "model13.keras": 0,
        "model14.keras": 0,
        "model15.keras": 0,
        "model16.keras": 0,
    }

    while episodes < 1000:
        max_reward = 0
        state_initial = env.reset()
        best_states = {}
        best_moves = {}

        max_reward, best_states, best_moves, best_model = model_use(
            env, state_initial.copy(), max_reward
        )
        if best_model is not None:
            model_names[best_model] += 1
            rewards[best_model] += max_reward

        if max_reward > 0:

            ##plot_staggered_outline_with_constraints(state_initial, move="initial")

            try:
                if best_moves["move_1"] is not None:
                    print(best_states["state_1"])
                    print()
                    print(best_moves["move_1"])
                    ##plot_staggered_outline_with_constraints(
                    ##    best_states["state_1"], best_moves[f"move_1"]
                    ##)
                try:
                    best_moves["move_2"] is not None
                except KeyError:
                    print("last_state")
                    pass
            except KeyError:
                pass

            try:
                if best_moves["move_2"] is not None:
                    print(best_states["state_2"])
                    print()
                    print(best_moves["move_2"])
                    ##plot_staggered_outline_with_constraints(
                    ##   best_states["state_2"], best_moves[f"move_2"]
                    ##)
                try:
                    best_moves["move_3"] is not None
                except KeyError:
                    print("last_state")
                    pass
            except KeyError:
                pass

            try:
                if best_moves["move_3"] is not None:
                    print(best_states["state_3"])
                    print()
                    print(best_moves["move_3"])
                    ##plot_staggered_outline_with_constraints(
                    ##    best_states["state_3"], best_moves[f"move_3"]
                    ##)
                    print("last_state")
            except KeyError:
                pass

            print(max_reward)

        else:

            ##plot_staggered_outline_with_constraints(state_initial, move="do nothing")

            print("Do nothing!")

        episodes += 1

    def print_model_stats(model_names, rewards, episodes):
        for model_name in model_names:
            print(f"{model_name}: {model_names[model_name]}")
            print(f"{model_name}: {rewards[model_name]}")
            if model_names[model_name] != 0:
                average_reward = rewards[model_name] / model_names[model_name]
                print(f"{model_name} average reward: {average_reward}\n")
            else:
                print(f"{model_name} average reward: N/A (no episodes)\n")
        print(f"Total episodes: {episodes}")

    print_model_stats(model_names, rewards, episodes)


if __name__ == "__main__":
    main()

import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from Env import CustomEnv, render
import Env as Env


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


def main():

    # Load the trained model
    model = load_model("model14.keras", custom_objects={"mse": MeanSquaredError()})
    remaining_moves = 3
    # Initialize the environment
    env = CustomEnv()

    # Reset the environment and preprocess the initial state
    state = env.reset()
    state = preprocess_state_with_priority(state, remaining_moves)

    done = False
    total_total_reward = 0

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

    # Specs to follow for anlaysis:
    start_time = time.time()
    test_episodes = 1000
    negative_cumulative_count = 0

    for episode in range(test_episodes):
        total_reward = 0

        state = env.reset()
        state = preprocess_state_with_priority(state, remaining_moves)
        done = False

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

                print(f"Move: {env.action_to_key(action)}, Reward: {reward}")

                remaining_moves -= 1

                next_state = preprocess_state_with_priority(next_state, remaining_moves)

                # Update state and total reward
                state = next_state
                total_reward += reward

                # Render the environment (optional)

        if total_reward < 0:
            negative_cumulative_count += 1

        total_total_reward += total_reward

    finish_time = time.time()

    print(f"Time taken for {test_episodes} episodes: {finish_time - start_time}")
    print(f"Average Reward: {total_total_reward/test_episodes}")
    print(f"Negative Reward Finish Count: {negative_cumulative_count}")
    print(
        f"Negative Reward Finish Percentage: {negative_cumulative_count/test_episodes}"
    )


if __name__ == "__main__":
    main()

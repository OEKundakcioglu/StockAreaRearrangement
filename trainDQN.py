import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

import time
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

from Env_wnr import (
    CustomEnv,
    render,
)  # Adjust this import to your custom environment

env = CustomEnv()

# Define the size of the input to the model based on your environment's state space

num_actions = 3721  # Including a 'do nothing' action


state_vector_size = 61 * 6  # Adjust this in your model definition


def build_model():
    model = Sequential(
        [
            Input(shape=(367,)),  # Input layer size based on the state vector size
            Dense(500, activation="tanh"),  # First hidden layer
            # Second hidden layer
            Dense(875, activation="tanh"),
            Dense(1750, activation="tanh"),  # Third hidden layer
            Dense(
                3721, activation="relu"
            ),  # Output layer size based on the number of actions
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.00001), loss="mse")
    return model


model = build_model()  # Policy Model
target_model = clone_model(model)  # Target Model
target_model.set_weights(model.get_weights())  # Target Model


def update_target_model():
    """Update the weights of the target network from the policy network."""
    target_model.set_weights(model.get_weights())


def preprocess_state_with_priority(state, remaining_moves, moves=8):
    """
    Converts the state into a flat vector with priorities encoded and blocked status.
    Each slot is represented by a 6-element vector.
    """
    processed_state = []
    for position_key in sorted(state.keys()):
        bobbin = state[position_key]
        if bobbin is None:
            processed_state.extend([0, 0, 0, 0, 0, 0])  # No bobbin and not blocked
        else:
            priority_encoding = [0, 0, 0, 0, 0]
            priority_encoding[bobbin.priority - 1] = 1  # 1,2,3,4,5
            is_blocked = 1 if bobbin.is_blocked_o else 0
            processed_state.extend(priority_encoding + [is_blocked])
    moves_normalized = (remaining_moves - 1) / (
        moves - 1
    )  # Normalizing based on max moves (3)
    processed_state.append(moves_normalized)
    return np.array(processed_state).reshape(1, -1)


def get_valid_actions():
    valid_actions = []
    valid_actions.append(3720)  # Add action number 3720 for "do nothing"
    for action in range(0, 3720):
        from_key, to_key = env.action_to_key(action)

        if env.is_move_valid(from_key, to_key):
            valid_actions.append(action)

    return valid_actions


def choose_action(state, valid_actions, epsilon, model, remaining_moves):
    if np.random.rand() <= epsilon:
        if np.random.uniform(0, 1) < 0.8:
            # Randomly select from valid actions
            return np.random.choice(valid_actions)  # explore
        else:
            if remaining_moves < 8:
                return 3720
            else:
                return np.random.choice(valid_actions)
    else:
        # Predict Q-values for the current state for all actions
        q_values = model.predict(state)
        # Filter Q-values for valid actions only and select the action with the highest Q-value
        q_values_filtered = q_values[0][valid_actions]
        return valid_actions[np.argmax(q_values_filtered)]  # exploit


def replay(memory, batch_size, policy_model, target_model, discount_rate):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        # Predict future rewards with target model
        future_rewards = target_model.predict(next_state)[0]
        # Current best Q-value (max Q-value for the next state)
        target_q_value = (
            reward if done else reward + discount_rate * np.max(future_rewards)
        )
        # Get current Q-values from the policy model and update the Q-value for the action taken
        current_q_values = policy_model.predict(state)
        current_q_values[0][action] = target_q_value
        # Fit the policy model
        policy_model.fit(state, current_q_values, epochs=1, verbose=0)


# Training Hyperparameters

episodes = 0
max_episodes = 400000  # to stop overfitting if the training runs too much
batch_size = 45
discount_rate = 0.99
update_target_network_every = 6
epsilon = 0.9
epsilon_min = 0.03
epsilon_decay = 0.9999
memory = deque(maxlen=800)

hours = 10  # set this to the number of hours you want to run the training can be as high as you want but as low as 3
hours_barrage = hours - 2

running = True

t0 = time.time()

# for e in range(episodes):
while running:
    episodes += 1
    remaining_moves = 8
    raw_state = env.reset()
    render(raw_state)
    state = preprocess_state_with_priority(raw_state, remaining_moves)

    done = False
    total_reward = 0

    rewards = []  # List for holding last 100 episodes rewards

    while not done:
        valid_actions = get_valid_actions()  # Get valid actions for the current state

        if env.perfect_scenerio():
            done = True
            print("Perfect Scenerio")
            action = 3720
            next_state, reward, done, _ = env.step(action, total_reward)

            remaining_moves -= 1

            next_state = preprocess_state_with_priority(next_state, remaining_moves)
        else:
            action = choose_action(
                state, valid_actions, epsilon, model, remaining_moves
            )  # Choose among valid actions
            next_state, reward, done, _ = env.step(action, total_reward)

            remaining_moves -= 1

            next_state = preprocess_state_with_priority(next_state, remaining_moves)

        total_reward += reward

        memory.append((state, action, reward, next_state, done))
        state = next_state

        t1 = time.time()
        total = t1 - t0

        if total > hours * 60 * 60:
            running = False
        if total > hours_barrage * 60 * 60:
            epsilon_min = 0.01
        if episodes > max_episodes:
            running = False

        if len(rewards) <= 100:
            rewards.append(total_reward)
        else:
            if len(rewards) > 100:
                rewards.pop(0)  # Remove the earliest reward
                rewards.append(total_reward)  # Append the current reward

                negative_rewards = sum(1 for reward in rewards if reward < 0)
                if negative_rewards / len(rewards) > 0.03:
                    running = True
                else:
                    running = False

        rewards.append(total_reward)  # Append the total reward to the rewards list

        if len(rewards) > 100:

            rewards = rewards[-100:]  # Keep only the last 100 episodes rewards
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if len(memory) > batch_size:
            replay(memory, batch_size, model, target_model, discount_rate)

    if episodes % update_target_network_every == 0:
        update_target_model()

    print(
        f"Episode: {episodes+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}"
    )

model.save("model16.keras")
print("Model saved")
print("Training Done")
print("Total Episodes: ", episodes)
print("Total Time: ", total)
print("You can find the model saved as 'model14.keras' in the current directory.")

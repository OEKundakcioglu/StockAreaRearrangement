import gym
import numpy as np
import random
from gym import spaces
from Bobbin import Bobbin

# Assuming Bobbin is defined elsewhere as:
# Bobbin = namedtuple('Bobbin', ['priority', 'is_blocked_o'])
# Make sure Bobbin class is properly defined and imported


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(
            3721
        )  # 3600 moves + 120 swaps +1 do nothing
        self.observation_space = spaces.Dict(
            {
                "positions": spaces.MultiDiscrete(
                    [6] * 60
                ),  # Priority levels for each position
                "movable": spaces.MultiBinary(60),  # Movability flag for each position
            }
        )
        self.illegal_actions = self.get_illegal_actions_from_file()
        self.state = None
        self.layer_configs = {
            "1st": {"dimensions": (5, 5), "check_columns": True},
            "2nd": {"dimensions": (5, 4), "check_columns": True},
            "3rd": {"dimensions": (5, 3), "check_columns": False},
            "swap": {"dimensions": (1, 1), "check_columns": False},
        }

    def valid_key(self, layer, r, c):
        """Check if the generated key is valid within the environment dimensions."""
        if r < 1 or c < 1:
            return False  # This check remains valid for all layers

        # Adjusting to check based on layer configurations
        if layer not in self.layer_configs:
            return False  # If the layer isn't defined in the configurations

        rows, cols = self.layer_configs[layer]["dimensions"]
        if r > rows or c > cols:
            return False  # If the specified row or column exceeds the layer's bounds

        return True

    def reset(self):
        self.state = self.initialize_state()
        self.total_movements = 0
        return self.state

    def initialize_state(self):
        detailed_state = {}
        priority_options = [1, 2, 3, 4, 5]
        first_layer_bobbin_c = int(round(random.uniform(15, 25)))
        f = first_layer_bobbin_c
        second_layer_bobbin_c = int(round(random.uniform(0, round(f))))
        s = second_layer_bobbin_c
        third_layer_bobbin_c = int(round(random.uniform(0, round(s))))
        t = third_layer_bobbin_c

        # Initialize all positions to None
        for layer, config in self.layer_configs.items():
            rows, cols = config["dimensions"]
            for row in range(1, rows + 1):
                for col in range(1, cols + 1):
                    key = f"{layer}_{row},{col}"
                    detailed_state[key] = None

        for layer, config in self.layer_configs.items():
            rows, cols = config["dimensions"]
            for row in range(1, rows + 1):
                for col in range(1, cols + 1):
                    key = f"{layer}_{row},{col}"
                    if layer == "swap":
                        detailed_state[key] = None
                    elif layer == "2nd":
                        if (
                            detailed_state[f"1st_{row},{col}"] is not None
                            and detailed_state[f"1st_{row},{col+1}"] is not None
                            and random.uniform(0, 1)
                            < (s / 4)  # lower this if you want more bobbins
                        ):
                            priority = random.choice(priority_options)
                            is_blocked_o = False
                            if config["check_columns"] and (col == 1 or col == cols):
                                is_blocked_o = random.choice([True, False])
                            detailed_state[key] = Bobbin(priority, is_blocked_o)
                            s -= 1
                    elif layer == "3rd":
                        if (
                            detailed_state[f"2nd_{row},{col}"] is not None
                            and detailed_state[f"2nd_{row},{col+1}"] is not None
                            and random.uniform(0, 1)
                            < (t / 2)  # lower this if you want more bobbins
                        ):
                            priority = random.choice(priority_options)
                            is_blocked_o = False
                            if config["check_columns"] and (col == 1 or col == cols):
                                is_blocked_o = random.choice([True, False])
                            detailed_state[key] = Bobbin(priority, is_blocked_o)
                    else:  # 1st layer
                        if random.uniform(0, 1) < (
                            f / 5  # lower this if you want more bobbins
                        ):  # Random choice to place a bobbin
                            priority = random.choice(priority_options)
                            is_blocked_o = False
                            if config["check_columns"] and (col == 1 or col == cols):
                                is_blocked_o = random.choice([True, False])
                            detailed_state[key] = Bobbin(priority, is_blocked_o)
                            f -= 1
                        else:
                            detailed_state[key] = None
        return detailed_state

    def step(self, action, previous_reward_cumulative):
        done = False

        # if (
        #    self.perfect_scenerio()
        # ):
        #    return self.state, 1000, True, {}

        reward = 0
        if action not in self.illegal_actions:
            from_key, to_key = self.action_to_key(
                action
            )  # Decode action to 'from' and 'to' keys
        else:
            print(f"Illegal action: {action}")
            return self.state, -1000, True, {}

        total_prio_before = self.calculate_total_priority_of_top_bobbins()

        self.move_bobbin(from_key, to_key)
        self.total_movements += 1
        reward = self.calculate_reward(
            total_prio_before, action, previous_reward_cumulative, self.total_movements
        )
        if self.total_movements >= 3:
            done = True
        if (
            self.perfect_scenerio() or action == 3720
        ):  # Check if the episode has reached its end condition
            done = True

        print(f"Total movements: {self.total_movements}")

        return self.state, reward, done, {}

    def move_bobbin(self, from_key, to_key):
        if not self.is_move_valid(from_key, to_key):
            # print(f"Invalid move from {from_key} to {to_key}")
            return False

        # If move involves the swap, special handling is required
        if "swap" in from_key or "swap" in to_key:

            bobbin = self.state[from_key]
            self.state[to_key] = bobbin
            self.state[from_key] = None

            # Swap logic here: move bobbin to/from swap
            # Example: self.state[to_key] = self.state[from_key]; self.state[from_key] = None
            # If some other logic is required for swap, implement it here
            print(f"Moved bobbin from {from_key} to {to_key}")
        else:
            # Regular move: Update the state with the new bobbin positions
            bobbin = self.state[from_key]
            self.state[to_key] = bobbin
            self.state[from_key] = None
            print(f"Moved bobbin from {from_key} to {to_key}")

        return True

    def get_illegal_actions_from_file(self):
        with open("illegal_actions.txt", "r") as f:
            illegal_actions = {int(line.strip()): True for line in f}
        return illegal_actions

    def is_move_valid(self, from_key, to_key):

        st_layer_counter = 0

        for key, value in self.state.items():
            if value is not None and "1st" in key:
                st_layer_counter += 1

        if from_key not in self.state or to_key not in self.state:
            # print(f"Invalid keys: {from_key}, {to_key}")
            return False  # Invalid keys

        from_bobbin = self.state[from_key]
        to_bobbin = self.state[to_key]

        # Check if move is valid based on the environment's rules
        # Example: Check if 'from' slot has a bobbin and 'to' slot is empty or valid for swap
        # structure like this
        if from_bobbin is None or to_bobbin is not None:
            return False

        # Additional checks for move validity
        if "swap" in from_key and "swap" in to_key:
            # print("Invalid move: Cannot move from swap to swap")
            return False

        # if "1st" in from_key and "1st" in to_key:
        # experimental condition to make sure bobbin doesn't do ridiculus stuff,
        # for clarification for our group members this contrain is just used in the training of last 2 models model9 and model10
        #    return False

        # if (st_layer_counter < 20) and (
        #    ("1st" in from_key and "2nd" in to_key)
        #    or ("1st" in from_key and "3rd" in to_key)
        # ):
        # experimental condition to make sure bobbin doesn't do ridiculus stuff
        # for clarification for our group members this contrain is just used in the training of last 2 models model9 and model10
        #    return False

        if "swap" in from_key:
            # for clarification for our group members this contrain is just used in the training of last 2 models model9 and model10
            return False

        # if "1st" in from_key and "swap" in to_key:
        # second experimental
        # for clarification for our group members this contrain is just used in the training of last 2 models model9 and model10
        #    return False

        # if st_layer_counter >= 23:
        #    if "1st" in to_key:
        #        return False
        # for clarification for our group members this contrain is just used in the training of last 2 models model9 and model10

        if "swap" in from_key:
            # Moving from swap to a position
            if not self.is_bobbin_free(to_key):
                # print(f"Invalid move: {to_key} is not free or blocked")
                return False

            if not self.is_valid_supporting_slots(to_key) or self.does_logical_error(
                from_key, to_key
            ):
                # print(f"Invalid move: {to_key} does not have valid supporting slots")
                return False

            # Moving from position to position
            if from_key == to_key:
                # print("Invalid move: 'from' and 'to' positions cannot be the same")
                return False

        elif "swap" in to_key:
            # Moving from a position to swap
            if not self.is_bobbin_free(from_key):
                # print(f"Invalid move: {from_key} is not free or blocked")
                return False

            if not self.is_valid_supporting_slots(to_key) or self.does_logical_error(
                from_key, to_key
            ):
                # print(f"Invalid move: {to_key} does not have valid supporting slots")
                return False

            # Moving from position to position
            if from_key == to_key:
                # print("Invalid move: 'from' and 'to' positions cannot be the same")
                return False

        else:
            # Moving from position to position
            if from_key == to_key:
                # print("Invalid move: 'from' and 'to' positions cannot be the same")
                return False

            if not self.is_bobbin_free(from_key):
                # print(f"Invalid move: {from_key} is not free or blocked")
                return False

            if not self.is_valid_supporting_slots(to_key) or self.does_logical_error(
                from_key, to_key
            ):
                # print(f"Invalid move: {to_key} does not have valid supporting slots")
                return False

        return True

    def does_logical_error(self, from_key, to_key):
        # List of actions that always cause a logical error

        # Convert the error actions to a list of from_key and to_key pairs

        row, col = self.get_position_coordinates(from_key)

        if "1st" in from_key:
            if (
                self.valid_key("2nd", row, col - 1)
                and self.state.get(f"2nd_{row},{col-1}") == to_key
            ):
                return True
            if (
                self.valid_key("2nd", row, col)
                and self.state.get(f"2nd_{row},{col}") == to_key
            ):
                return True

        elif "2nd" in from_key:
            if (
                self.valid_key("3rd", row, col - 1)
                and self.state.get(f"3rd_{row},{col-1}") == to_key
            ):
                return True
            if (
                self.valid_key("3rd", row, col)
                and self.state.get(f"3rd_{row},{col}") == to_key
            ):
                return True

        return False

    def is_bobbin_free(self, position_key):
        """Check if the given position is free or blocked."""
        row, col = self.get_position_coordinates(position_key)

        if (
            self.state.get(position_key) is not None
            and self.state[position_key].is_blocked_o
        ):
            return False  # The position itself is blocked, thus not free.

        # Initialize variables to None to ensure they have a value even if not set later
        upper_left_key, upper_right_key = None, None

        if "1st" in position_key:
            if self.valid_key("2nd", row, col - 1):
                upper_left_key = f"2nd_{row},{col-1}"
            else:
                upper_left_key = None
            if self.valid_key("2nd", row, col):
                upper_right_key = f"2nd_{row},{col}"
            else:
                upper_right_key = None
        elif "2nd" in position_key:
            if self.valid_key("3rd", row, col - 1):
                upper_left_key = f"3rd_{row},{col-1}"
            else:
                upper_left_key = None
            if self.valid_key("3rd", row, col):
                upper_right_key = f"3rd_{row},{col}"
            else:
                upper_right_key = None

        # Ensure at least one of the upper positions is free if they exist
        upper_left_free = upper_left_key is None
        if upper_left_key is not None:
            upper_left_free = self.state.get(upper_left_key) is None

        upper_right_free = upper_right_key is None
        if upper_right_key is not None:
            upper_right_free = self.state.get(upper_right_key) is None

        return upper_left_free and upper_right_free

    def is_valid_supporting_slots(self, position_key):
        """Check if the supporting slots of the given position are valid."""
        row, col = self.get_position_coordinates(position_key)
        validity = False
        # For positions on the 1st layer, they are always valid as they have no supporting slots below.
        if "1st" in position_key:
            validity = True
        # For positions on the 2nd layer, check if corresponding slots on the 1st layer are not empty.
        elif "2nd" in position_key:
            supporting_keys = [f"1st_{row},{col}", f"1st_{row},{col+1}"]
            first = self.state.get(supporting_keys[0])
            second = self.state.get(supporting_keys[1])
            if first is not None and second is not None:
                validity = True
        # For positions on the 3rd layer, check if corresponding slots on the 2nd layer are not empty.
        elif "3rd" in position_key:
            supporting_keys = [f"2nd_{row},{col}", f"2nd_{row},{col+1}"]
            first = self.state.get(supporting_keys[0])
            second = self.state.get(supporting_keys[1])
            if first is not None and second is not None:
                validity = True

        # Check if all supporting positions exist and are not empty.
        return validity

    def get_position_coordinates(self, position_key):
        """Get the row and column coordinates from the position key."""
        row, col = map(int, position_key.split("_")[1].split(","))
        return int(row), int(col)

        # if the move is from swap to a position, the position should be free and not blocked
        # if the move is from a position to swap, the swap should be free and not blocked
        # if the move is from a position to a position, the position should be free and not blocked
        # if the move is from position to position, from position has to contain a bobbin. from and to cannot be same.
        # if the move is from swap to swap, it is not valid.
        # if the move is from position to position, to position it cannot support itself so it should have both bobbins under after the move this is the most common mistake.

    def action_to_key(self, action):
        """Converts an action number into the corresponding dictionary keys for 'from' and 'to' positions."""
        total_positions = 60  # Total positions in the grid
        swap_key = "swap_1,1"  # Assuming a single swap position key

        if action < 3600:
            # Regular moves within the grid
            from_index = action // 60
            to_index = action % 60
            from_key = self.index_to_key(from_index)
            to_key = self.index_to_key(to_index)
        elif 3600 <= action < 3720:
            # Swap actions
            swap_action_index = action - 3600
            if swap_action_index < total_positions:
                # Moving from a grid position to the swap
                grid_position_index = swap_action_index
                from_key = self.index_to_key(grid_position_index)
                to_key = swap_key
            else:
                # Moving from the swap to a grid position
                grid_position_index = swap_action_index - total_positions
                from_key = swap_key
                to_key = self.index_to_key(grid_position_index)
        else:
            # Do nothing action
            return None, None

        return from_key, to_key

    def index_to_key(self, index):
        """Convert a grid position index to its corresponding dictionary key, adjusting for layer-specific dimensions."""
        if 0 <= index <= 24:  # First layer
            layer = "1st"
            row, col = divmod(index, 5)
        elif 25 <= index <= 44:  # Second layer
            layer = "2nd"
            index -= 25  # Adjust index to start from 0 for the second layer
            row, col = divmod(index, 4)
        elif 45 <= index <= 59:  # Third layer
            layer = "3rd"
            index -= 45  # Adjust index to start from 0 for the third layer
            row, col = divmod(index, 3)
        else:
            raise ValueError(f"Index {index} is out of bounds for the grid.")

        row += 1  # Adjust row and column to be 1-based
        col += 1
        return f"{layer}_{row},{col}"

    def calculate_reward(
        self, total_prio_before, action, previous_reward_cumulative, total_movements
    ):
        upper_bound = 23
        lower_bound = 15  # bobbin count in the 1st layer for 5x5 case

        st_layer_counter = 0

        for key, value in self.state.items():
            if value is not None and "1st" in key:
                st_layer_counter += 1

        reward = 0
        penalize = True

        action_from, action_to = self.action_to_key(action)

        if action == 3720 and previous_reward_cumulative > 0:
            reward += 11
            penalize = False
        elif action == 3720 and previous_reward_cumulative < 0:
            reward -= 100
            penalize = False

        if action_from is not None and action_to is not None:
            if "1st" in action_from and "1st" in action_to:
                reward -= 15

            if st_layer_counter >= upper_bound:
                if "1st" in action_to:
                    reward -= 10

                if "1st" in action_from and "2nd" in action_to:
                    reward += 15

                if "1st" in action_from and "3rd" in action_to:
                    reward += 15

            if st_layer_counter < lower_bound:
                if "1st" not in action_from and "1st" in action_to:
                    reward += 5

                if "1st" in action_from and "2nd" in action_to:
                    reward -= 15

                if "1st" in action_from and "3rd" in action_to:
                    reward -= 15

        if action == 3720:
            penalize = False

        if st_layer_counter < upper_bound or st_layer_counter >= lower_bound:
            if self.perfect_scenerio() == True:
                reward += 200
                penalize = False

        if self.rescued(total_prio_before) > 0:
            reward += self.rescued(total_prio_before)
            penalize = False

        if self.infeasible(self.state) == True:
            reward -= 1500
            return reward

        if penalize:
            reward -= (
                5 * total_movements
            )  # Penelize Each move to ensure minimum movement is targeted.
        # Assume each action gives a fixed reward; adjust as needed
        return reward

    def infeasible(self, state):
        # Connect to feasibility check code to compare our state with the code and return the result for the feasibility check. If the state is infeasible return True else return False.
        # use only if you need to train with these rules intact.
        return False

    def rescued(self, total_prio_before):

        total_prio_after = self.calculate_total_priority_of_top_bobbins()

        if total_prio_after > total_prio_before:
            dif = total_prio_after - total_prio_before
            if dif >= 6:
                return 34
            elif dif >= 5:
                return 23
            elif dif >= 4:
                return 12
            elif dif >= 3:
                return 5
            elif dif >= 2:
                return 2
            elif dif >= 1:
                return 1

        return 0

    def calculate_top_bobbins(self):
        top_bobbins = []
        for key, value in self.state.items():

            if value is not None and "3rd" in key:
                top_bobbins.append(key)

            if value is not None and "1st" in key:  # nd value.is_blocked_o == False:
                row, col = map(int, key.split("_")[1].split(","))

                first_check = True
                second_check = True
                third_check = True
                fourth_check = True
                fifth_check = True

                if self.valid_key("2nd", row, col - 1):
                    if self.state.get(f"2nd_{row},{col-1}") is None:
                        first_check = True
                    else:
                        first_check = False

                if self.valid_key("2nd", row, col):
                    if self.state.get(f"2nd_{row},{col}") is None:
                        second_check = True
                    else:
                        second_check = False

                if self.valid_key("3rd", row, col):
                    if self.state.get(f"3rd_{row},{col}") is None:
                        third_check = True
                    else:
                        third_check = False

                if self.valid_key("3rd", row, col - 1):
                    if self.state.get(f"3rd_{row},{col-1}") is None:
                        fourth_check = True
                    else:
                        fourth_check = False

                if self.valid_key("3rd", row, col - 2):
                    if self.state.get(f"3rd_{row},{col-2}") is None:
                        fifth_check = True
                    else:
                        fifth_check = False

                if (
                    first_check
                    and second_check
                    and third_check
                    and fourth_check
                    and fifth_check
                ):
                    top_bobbins.append(key)

            if value is not None and "2nd" in key and value.is_blocked_o == False:
                row, col = map(int, key.split("_")[1].split(","))
                first_check = True
                second_check = True

                if self.valid_key("3rd", row, col - 1):
                    if self.state.get(f"3rd_{row},{col-1}") is None:
                        first_check = True
                    else:
                        first_check = False

                if self.valid_key("3rd", row, col):
                    if self.state.get(f"3rd_{row},{col}") is None:
                        second_check = True
                    else:
                        second_check = False

                if first_check and second_check:
                    top_bobbins.append(key)

        return top_bobbins

    def calculate_total_priority_of_top_bobbins(self):

        top_bobbins = self.calculate_top_bobbins()

        total_priority = 0

        for key in top_bobbins:
            total_priority += self.state[key].priority

        return total_priority

    def perfect_scenerio(self):
        terminate = False
        top_bobbins = self.calculate_top_bobbins()

        for key in top_bobbins:
            row, col = map(int, key.split("_")[1].split(","))
            blocked_bobbinsP = []

            # Check for layer 1
            if "2nd" in key or "3rd" in key:
                if (
                    self.valid_key("1st", row, col)
                    and self.state.get(f"1st_{row},{col}") is not None
                ):
                    blocked_bobbinsP.append(self.state.get(f"1st_{row},{col}").priority)
                if (
                    self.valid_key("1st", row, col + 1)
                    and self.state.get(f"1st_{row},{col+1}") is not None
                ):
                    blocked_bobbinsP.append(
                        self.state.get(f"1st_{row},{col+1}").priority
                    )
                if "3rd" in key:
                    if (
                        self.valid_key("1st", row, col + 2)
                        and self.state.get(f"1st_{row},{col+2}") is not None
                    ):
                        blocked_bobbinsP.append(
                            self.state.get(f"1st_{row},{col+2}").priority
                        )

            # Check for layer 2
            if "3rd" in key:
                if (
                    self.valid_key("2nd", row, col)
                    and self.state.get(f"2nd_{row},{col}") is not None
                ):
                    blocked_bobbinsP.append(self.state.get(f"2nd_{row},{col}").priority)
                if (
                    self.valid_key("2nd", row, col + 1)
                    and self.state.get(f"2nd_{row},{col+1}") is not None
                ):
                    blocked_bobbinsP.append(
                        self.state.get(f"2nd_{row},{col+1}").priority
                    )

            for priorityU in blocked_bobbinsP:
                if priorityU >= self.state[key].priority:
                    terminate = False
                    return terminate
                else:
                    terminate = True

        is_f = True

        for key in top_bobbins:
            if "2nd" in key or "3rd" in key:
                is_f = False

        if is_f:
            terminate = True

        # Implement logic to determine if the episode has ended
        return terminate  # Example condition


def render(state):
    # can be improved to show the grid in a more readable format maybe 3d even visual.
    print("Current State:")
    if state is not None:
        for key, value in state.items():
            priority = None
            is_blocked_o = None
            if value is not None:
                priority = state.get(key).priority
                is_blocked_o = state.get(key).is_blocked_o

            print(f"{key}: {priority} {is_blocked_o}")

    else:
        print("State not initialized.")


# Make sure to define or import Bobbin correctly before using this environment

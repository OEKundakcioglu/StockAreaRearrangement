illegal_actions = []

# Define the grid dimensions
layer_configs = {
    "1st": {"dimensions": (5, 5), "check_columns": True},
    "2nd": {"dimensions": (5, 4), "check_columns": True},
    "3rd": {"dimensions": (5, 3), "check_columns": False},
}

layer_start_indices = {
    "1st": 0,
    "2nd": 25,
    "3rd": 45,
}


def index_to_position(index):
    """Converts a linear index to a (layer, row, column) position."""
    if index < 25:
        layer = "1st"
        index_in_layer = index
    elif 25 <= index < 45:
        layer = "2nd"
        index_in_layer = index - 25
    elif 45 <= index < 60:
        layer = "3rd"
        index_in_layer = index - 45
    else:
        raise ValueError("Index out of range")

    rows, columns = layer_configs[layer]["dimensions"]
    row = index_in_layer // columns + 1
    column = index_in_layer % columns + 1

    return (layer, row, column)


def position_to_index(layer, row, column):
    """Converts a (layer, row, column) position to a linear index."""
    rows, columns = layer_configs[layer]["dimensions"]
    layer_start = layer_start_indices[layer]
    index_in_layer = (row - 1) * columns + (column - 1)
    return layer_start + index_in_layer


# Iterate through all possible actions
for action in range(3600):
    from_index = action // 60
    to_index = action % 60
    from_position = index_to_position(from_index)
    to_position = index_to_position(to_index)

    from_layer, from_row, from_column = from_position
    to_layer, to_row, to_column = to_position

    # Check if move is from a lower layer to the layer above
    if from_layer == "1st" and to_layer == "2nd":
        if (
            (from_row == to_row and from_column == to_column)
            or (from_row == to_row and from_column == to_column - 1)
            or (from_row == to_row and from_column == to_column + 1)
            or (from_row == to_row - 1 and from_column == to_column)
            or (from_row == to_row - 1 and from_column == to_column + 1)
            or (from_row == to_row - 1 and from_column == to_column - 1)
        ):
            illegal_actions.append(action)
    elif from_layer == "2nd" and to_layer == "3rd":
        if (
            (from_row == to_row and from_column == to_column)
            or (from_row == to_row and from_column == to_column - 1)
            or (from_row == to_row and from_column == to_column + 1)
            or (from_row == to_row - 1 and from_column == to_column)
            or (from_row == to_row - 1 and from_column == to_column + 1)
            or (from_row == to_row - 1 and from_column == to_column - 1)
        ):
            illegal_actions.append(action)

with open("illegal_actions.txt", "w") as f:
    for action in illegal_actions:
        f.write(f"{action}\n")

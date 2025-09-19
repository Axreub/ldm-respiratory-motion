import json
import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()


def load_paths(paths_file: str = "paths.json") -> dict:
    """
    Load the paths configuration from a JSON file located at the base path.

    Args:
        paths_file (str): The filename of the JSON file containing path definitions.
                          Defaults to "paths.json".

    Returns:
        dict: The parsed dictionary of all paths from the JSON file.
    """
    BASE_PATH = os.getenv("BASE_PATH")
    location_path = os.path.join(BASE_PATH, paths_file)

    if not os.path.exists(location_path):
        raise FileNotFoundError(f"Paths file '{paths_file}' not found.")

    with open(location_path, "r") as file:
        paths = json.load(file)
    return paths


def construct_paths(
    data: Any,
    is_input_path: bool = False,
    is_model_path: bool = False,
    category: str = None,
) -> dict:
    """
    Recursively update all string paths in a nested dictionary or list to include the appropriate base path.

    For each string value:
      - If it is a model path, leave it unchanged.
      - Otherwise, prepend the STORAGE_BASE_PATH environment variable.

    Args:
        data (Any): The data structure (dict, list, or str) containing paths to update.
        is_input_path (bool): Whether the current value is an input data path.
        is_model_path (bool): Whether the current value is a model path.

    Returns:
        dict, list, or str: The data structure with all string paths updated to include the appropriate base path.
    """
    if isinstance(data, dict):
        return {
            key: construct_paths(
                value,
                is_input_path=(key == "input_data_path"),
                is_model_path=(key == "model_path"),
                category=category,
            )
            for key, value in data.items()
        }
    elif isinstance(data, str):
        if is_model_path:
            return data
        else:
            return os.path.join(os.getenv("STORAGE_BASE_PATH", ""), data)
    return data  # If it's not a dict or str, return as-is.


def get_paths(category: str, paths_file: str = "paths.json") -> dict:
    """
    Retrieve a dictionary of constructed paths for a specific category from the paths file.

    This function loads the paths configuration, selects the specified category,
    and updates all string paths to include the appropriate base path (STORAGE_BASE_PATH).

    Args:
        category (str): The top-level key in the paths file to retrieve (e.g., "autoencoder").
        paths_file (str): The filename of the JSON file containing path definitions.
                          Defaults to "paths.json".

    Returns:
        dict: A dictionary of paths for the specified category, with all string paths
              updated to include the appropriate base path.
    """
    paths = load_paths(paths_file)

    if category not in paths:
        raise KeyError(f"Category '{category}' not found in paths file.")

    return construct_paths(paths[category], category=category)

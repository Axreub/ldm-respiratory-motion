#!/bin/bash

# Set the base path to the directory containing this script
BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ENV_FILE=".env"
VENV_PATH="$BASE_PATH/.venv"

echo "BASE_PATH=$BASE_PATH" > "$ENV_FILE"
echo "STORAGE_BASE_PATH=$BASE_PATH" >> "$ENV_FILE"


if [ -f "$ENV_FILE" ]; then
    echo "BASE_PATH written to $ENV_FILE: $BASE_PATH"
else
    echo "Failed to create or write to $ENV_FILE."
    exit 1
fi

# Copy template files if they don't exist
declare -a templates=("$PATHS_TEMPLATE" "$DIFF_ARGS_TEMPLATE" "$ENC_ARGS_TEMPLATE")
declare -a targets=("$PATHS_FILE" "$DIFF_ARGS_FILE" "$ENC_ARGS_FILE")

for i in "${!templates[@]}"; do
    template="${templates[$i]}"
    target="${targets[$i]}"
    if [ ! -f "$target" ]; then
        if [ -f "$template" ]; then
            cp "$template" "$target"
            echo "Copied $template to $target."
        else
            echo "Error: $template not found"
            exit 1
        fi
    else
        echo "$target already exists. Skipping copy."
    fi
done

VENV_PATH="$BASE_PATH/.venv"
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    echo "Created venv at $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed."
echo "To activate the virtual environment later: source .venv/bin/activate"
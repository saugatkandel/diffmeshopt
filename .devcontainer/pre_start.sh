#!/bin/bash

# --- 1. VS Code Extension Installation ---
# These are pulled directly from your devcontainer extensions list
extensions=(
    "ms-python.python"
    "ms-python.pylance"
    "charliermarsh.ruff"
    "tamasfe.even-better-toml"
    "google.gemini-vscode"
    "ms-toolsai.jupyter"
    "usernamehw.errorlens"
)

echo "Installing VS Code extensions..."
for ext in "${extensions[@]}"; do
    # Installing to the default root server directory
    code-server --install-extension "$ext" --extensions-dir /root/.vscode-server/extensions
done

# --- 2. Pixi Environment Setup (postCreateCommand) ---
# This runs the 'pixi install' you had in your config
if [ -d "/workspace" ]; then
    cd /workspace
    if [ -f "pixi.toml" ]; then
        echo "Found pixi.toml, running pixi install..."
        pixi install
    else
        echo "No pixi.toml found in /workspace. Skipping install."
    fi
fi

# --- 3. Shell Customization ---
# Adding the python interpreter path to the system PATH so it's the default
echo 'export PATH="/workspace/.pixi/envs/default/bin:$PATH"' >> ~/.bashrc
echo 'alias code="code-server"' >> ~/.bashrc

echo "Pre-start configuration complete."
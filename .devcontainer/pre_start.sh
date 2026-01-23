#!/bin/bash

# --- 1. Pixi & Kernel Registration (Persistent Workspace) ---
PIXI_ENV_PATH="/workspace/diffmeshopt/.pixi/envs/default"

if [ -d "/workspace/diffmeshopt" ]; then
    cd /workspace/diffmeshopt
    if [ -f "pixi.toml" ]; then
        echo "Running pixi install in /workspace/diffmeshopt..."
        pixi install --skip pytorch3d
        #pixi install
    else
        echo "No pixi.toml found in /workspace/diffmeshopt. Skipping install."
    fi
fi


# --- 3. Shell Customization ---
# Adding the python interpreter path to the system PATH so it's the default
echo 'export PATH="/workspace/diffmeshopt/.pixi/envs/default/bin:$PATH"' >> ~/.bashrc

echo "Pre-start configuration complete."
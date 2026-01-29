#!/bin/bash

# --- 1. Pixi & Kernel Registration (Persistent Workspace) ---

if [ -d "/workspace/diffmeshopt" ]; then
    cd /workspace/diffmeshopt
    if [ -f "pixi.toml" ]; then
        echo "Running pixi install in /workspace/diffmeshopt..."
        echo "Installing environments, skipping pytorch3d compilation..."
        pixi install --all --frozen --skip pytorch3d
    else
        echo "No pixi.toml found in /workspace/diffmeshopt. Skipping install."
    fi
fi


echo "Pre-start configuration complete."